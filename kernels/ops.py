"""Kernel-backed arithmetic for psgd_pro_fast (KronWhiten(ops=make_ops(flags))), registered as custom ops so that
torch.compile(fullgraph=True) can call them. Flags:
  k1   norm lower bounds of term1 and R with the triangle kernel K1 (CUDA)
  k2   term1 and P_i = Q_i^T Q_i with the Triton SYRK K2, mirrored into full matrices
  k2t  the same with K2 v2 (triangular grid; X X^T via a transpose and the X^T X kernel), from n = 1024
  k2c  the same with cublasSsyrk, mirrored with torch ops
  k2b  the same with a blocked SYRK of torch.matmul calls on the upper block pairs (kernels/k2.py), mirrored
  tri  term1 stays upper-triangle only (K2 without mirror) and is read only by K1 and cublasSsymm (needs k1)
  nan  with tri: fill term1's lower triangle with NaN (proves nothing reads it)
  all  use the kernels at every eligible size instead of only above the microbenchmark crossover
  packed (2-D path) term1 and P stored as packed upper triangles (K2 v2 writes them; consumers unpack, needs k2t)
Kernels apply to 2-D factors with n % 64 == 0 and n >= MIN_N (results/k1.json, k2.json: K1 never beat cuBLAS on
V @ A, its fused skew bound did at n >= 4096; K2 beat matmul at n >= 2048); other shapes keep the eager arithmetic."""
import torch
import triton
from torch import Tensor
from torch.library import custom_op, triton_op, wrap_triton
import psgd_pro_fast as fast
from kernels import blas as _blas
from kernels import k1 as _k1
from kernels.k2 import _syrk2_kernel, _syrk_kernel, _unpack_kernel, syrk_blocked_mirror


@custom_op("psgd::tri_vmm", mutates_args=())
def tri_vmm(V: Tensor, A: Tensor, sign: int) -> Tensor:
    return _k1.tri_vmm(V.contiguous(), A, sign)


@tri_vmm.register_fake
def _(V, A, sign):
    return V.new_empty(V.shape)


@custom_op("psgd::tri_rownorm2", mutates_args=())
def tri_rownorm2(A: Tensor) -> tuple[Tensor, Tensor]:
    r, m = _k1.tri_rownorm2(A)
    return r, m.clone()


@tri_rownorm2.register_fake
def _(A):
    return A.new_empty(A.shape[0]), A.new_empty(())


@custom_op("psgd::symm", mutates_args=())
def symm(S: Tensor, B: Tensor) -> Tensor:
    return _blas.symm(S, B)


@symm.register_fake
def _(S, B):
    return B.new_empty(S.shape[0], B.shape[1])


@custom_op("psgd::syrk_cublas", mutates_args=())
def syrk_cublas(A: Tensor, trans: bool) -> Tensor:
    return _blas.syrk(A, trans)


@syrk_cublas.register_fake
def _(A, trans):
    n = A.shape[1] if trans else A.shape[0]
    return A.new_empty(n, n)


@triton_op("psgd::syrk", mutates_args={})
def syrk(A: Tensor, trans: bool, mirror: bool) -> Tensor:
    if trans:
        (K, N), (s_k, s_n) = A.shape, A.stride()
    else:
        (N, K), (s_n, s_k) = A.shape, A.stride()
    C = torch.empty(N, N, device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BM"]), triton.cdiv(N, meta["BM"]))
    wrap_triton(_syrk_kernel)[grid](A, C, N, K, s_n, s_k, TRANS=trans, MIRROR=mirror, KC=128)
    return C


@triton_op("psgd::syrk2", mutates_args={})
def syrk_v2(A: Tensor, trans: bool, mirror: bool) -> Tensor:
    """K2 v2 (kernels/k2.py syrk2): X X^T as Y^T Y with Y = A^T copied contiguous."""
    if not trans:
        A = A.T.contiguous()
    (K, N), (s_k, s_n) = A.shape, A.stride()
    C = torch.empty(N, N, device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BM"]) * (triton.cdiv(N, meta["BM"]) + 1) // 2,)
    wrap_triton(_syrk2_kernel)[grid](A, C, N, K, s_n, s_k, TRANS=True, MIRROR=mirror, KC=128, PACKED=False)
    return C


@custom_op("psgd::symm_right", mutates_args=())
def symm_right(B: Tensor, S: Tensor) -> Tensor:
    return _blas.symm_right(B, S)


@symm_right.register_fake
def _(B, S):
    return B.new_empty(B.shape[0], S.shape[0])


@triton_op("psgd::syrk2p", mutates_args={})
def syrk_v2_packed(A: Tensor, trans: bool) -> Tensor:
    """K2 v2 writing the row-major packed upper triangle (n(n+1)/2 values)."""
    if not trans:
        A = A.T.contiguous()
    (K, N), (s_k, s_n) = A.shape, A.stride()
    C = torch.empty(N * (N + 1) // 2, device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BM"]) * (triton.cdiv(N, meta["BM"]) + 1) // 2,)
    wrap_triton(_syrk2_kernel)[grid](A, C, N, K, s_n, s_k, TRANS=True, MIRROR=False, KC=128, PACKED=True)
    return C


@triton_op("psgd::unpack", mutates_args={})
def unpack(Pk: Tensor, n: int) -> Tensor:
    C = torch.empty(n, n, device=Pk.device, dtype=Pk.dtype)
    wrap_triton(_unpack_kernel)[(triton.cdiv(n, 64), triton.cdiv(n, 64))](Pk, C, n, BM=64)
    return C


def side(S):
    """n of a symmetric matrix in storage form (square, or packed with n(n+1)/2 values)."""
    return S.shape[0] if S.dim() == 2 else int(((8 * S.shape[0] + 1) ** 0.5 - 1) / 2)


def nlb_tri(A, V, sign):
    """kernels.k1.norm_lower_bound_tri on the custom ops (traceable)."""
    A, flip = (A, 1) if A.is_contiguous() else (A.T, sign)
    sn = torch.finfo(A.dtype).smallest_normal
    rn2, amax = tri_rownorm2(A)
    nf = (A.diagonal().amax() if sign > 0 else amax) + sn
    Aj = (flip * _k1.sym_row(A, torch.argmax(rn2), sign) / nf).unsqueeze(0)
    V = Aj + torch.sgn(torch.sum(Aj * V, dim=1, keepdim=True)) * V
    for _ in range(2):
        V = flip * tri_vmm(V, A, sign) / nf
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + sn
        V = flip * tri_vmm(V, A, sign) / nf
    return nf * torch.amax(torch.linalg.vector_norm(V, dim=1))


MIN_N = dict(k1=4096, k2=2048, k2t=1024)


def make_ops(flags):
    flags = set(flags)
    assert "tri" not in flags or "k1" in flags, "tri needs k1 (term1's only other reader)"
    lo = {k: 64 if "all" in flags else v for k, v in MIN_N.items()}
    if "k2t" in flags:
        lo["k2"] = lo["k2t"]
    if "tri" in flags:  # term1's producer and both readers must switch together
        lo["k1"] = lo["k2"] = min(lo["k1"], lo["k2"])
    ok1 = lambda n: n % 64 == 0 and n >= lo["k1"]
    ok2 = lambda n: n % 64 == 0 and n >= lo["k2"]

    class KernelOps(fast.EagerOps):
        pass

    if "k1" in flags:
        KernelOps.norm_lower_bound_spd = staticmethod(
            lambda A, V: nlb_tri(A, V, 1) if ok1(A.shape[0]) else fast.norm_lower_bound_spd(A, V))
        KernelOps.norm_lower_bound_skh = staticmethod(
            lambda A, V: nlb_tri(A, V, -1) if ok1(A.shape[0]) else fast.norm_lower_bound_skh(A, V))
    if "k2t" in flags:
        sy = lambda X, trans: syrk_v2(X, trans, True)
    elif "k2c" in flags:
        sy = lambda X, trans: (lambda C: torch.triu(C) + torch.triu(C, 1).T)(syrk_cublas(X, trans))
    elif "k2b" in flags:
        sy = lambda X, trans: syrk_blocked_mirror(X, trans)
    else:
        sy = lambda X, trans: syrk(X, trans, True)
    if flags & {"k2", "k2t", "k2c", "k2b"}:
        KernelOps.gram_q = staticmethod(
            lambda q: sy(q, True) if ok2(q.shape[0]) else torch.tensordot(q, q, dims=([0], [0])))

        def syrk_full(X, trans):
            """Full symmetric X^T X (trans) or X X^T. K2 is ~1.7x faster on X^T X than on X X^T (results/k2.json):
            X X^T is computed as Y^T Y with Y = X^T copied contiguous (one O(n^2) memory pass)."""
            if not ok2(X.shape[1] if trans else X.shape[0]):
                return X.T @ X if trans else X @ X.T
            if trans or flags & {"k2t", "k2c", "k2b"}:
                return sy(X, trans)
            return sy(X.T.contiguous(), True)
        KernelOps.syrk = KernelOps.sym = staticmethod(syrk_full)
    if "tri" in flags or "packed" in flags:  # B12: term1 and P keep one triangle (square) or a packed triangle
        def sym_store(X, trans):
            n = X.shape[1] if trans else X.shape[0]
            if not ok2(n):
                return X.T @ X if trans else X @ X.T
            if "packed" in flags:
                return syrk_v2_packed(X, trans)
            C = syrk_v2(X, trans, False)
            if "nan" in flags:
                C = C.masked_fill(torch.ones_like(C, dtype=torch.bool).tril(-1), float("nan"))
            return C
        full = lambda S: unpack(S, side(S)) if S.dim() == 1 else S
        tri_ok = lambda S: "tri" in flags and S.dim() == 2 and ok2(S.shape[0])
        KernelOps.sym = staticmethod(sym_store)
        KernelOps.symm = staticmethod(lambda S, X: symm(S, X) if tri_ok(S) else full(S) @ X)
        KernelOps.symm_r = staticmethod(lambda X, S: symm_right(X, S) if tri_ok(S) else X @ full(S))
        spd = KernelOps.norm_lower_bound_spd
        KernelOps.norm_lower_bound_spd = staticmethod(lambda A, V: spd(full(A), V))
    if flags & {"k2", "k2t", "k2c", "k2b", "tri"}:
        def gram(Pg, i, expr):
            n = Pg.shape[i] if Pg.dim() == 2 else 0
            if not ok2(n):
                return expr(Pg, Pg)
            if "tri" not in flags:
                return sy(Pg, i == 1)
            C = syrk(Pg, i == 1, False)
            if "nan" in flags:
                C = C.masked_fill(torch.ones_like(C, dtype=torch.bool).tril(-1), float("nan"))
            return C
        KernelOps.gram = staticmethod(gram)
    if "tri" in flags:
        KernelOps.symm = staticmethod(lambda A, q: symm(A, q) if ok2(A.shape[0]) else A @ q)
    KernelOps.flags = frozenset(flags)
    return KernelOps
