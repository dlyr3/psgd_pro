"""K2: SYRK in Triton. C = X X^T for X = A (n x k) or X = A^T (A is k x n), computing only the upper-triangle output
tiles; tl.dot with input_precision="ieee" (exact fp32 products on FMA units; Triton's fp32 default is TF32).
The k loop accumulates KC-wide partial sums and adds them to the output accumulator: a plain k-order FMA chain
(what cublasSsyrk does, bitwise) has about twice cuBLAS GEMM's error against fp64.
mirror=True also writes each off-diagonal tile transposed below the diagonal, so C is a full symmetric matrix."""
import torch
import triton
import triton.language as tl


def _configs():
    return [triton.Config({"BM": bm, "BK": bk}, num_warps=w, num_stages=s)
            for bm, bk, w, s in [(64, 32, 4, 3), (64, 16, 4, 4), (64, 64, 4, 2), (128, 16, 8, 3), (128, 32, 8, 2),
                                 (128, 16, 4, 3), (64, 32, 8, 3)]]


@triton.autotune(configs=_configs(), key=["N", "K", "TRANS", "MIRROR"])
@triton.jit
def _syrk_kernel(A, C, N, K, s_n, s_k, TRANS: tl.constexpr, MIRROR: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                 KC: tl.constexpr):
    ti, tj = tl.program_id(0), tl.program_id(1)
    if ti > tj:
        return
    ri = ti * BM + tl.arange(0, BM)
    rj = tj * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for kc in range(0, K, KC):
        part = tl.zeros((BM, BM), dtype=tl.float32)
        for k0 in range(kc, kc + KC, BK):
            kk = k0 + rk
            x = tl.load(A + ri[:, None] * s_n + kk[None, :] * s_k, mask=(ri[:, None] < N) & (kk[None, :] < K), other=0.0)
            y = tl.load(A + kk[:, None] * s_k + rj[None, :] * s_n, mask=(kk[:, None] < K) & (rj[None, :] < N), other=0.0)
            part = tl.dot(x, y, part, input_precision="ieee")
        acc += part
    tl.store(C + ri[:, None] * N + rj[None, :], acc, mask=(ri[:, None] < N) & (rj[None, :] < N))
    if MIRROR:
        if ti != tj:
            tl.store(C + rj[:, None] * N + ri[None, :], tl.trans(acc), mask=(rj[:, None] < N) & (ri[None, :] < N))


def _configs2():
    return [triton.Config({"BM": bm, "BK": bk}, num_warps=w, num_stages=s)
            for bm, bk, w, s in [(64, 32, 4, 2), (64, 32, 4, 3), (64, 16, 4, 4), (128, 32, 8, 3), (128, 16, 8, 3),
                                 (128, 16, 8, 5), (128, 32, 8, 2), (64, 64, 4, 2), (128, 64, 8, 2)]]


@triton.autotune(configs=_configs2(), key=["N", "K", "TRANS", "MIRROR"])
@triton.jit
def _syrk2_kernel(A, C, N, K, s_n, s_k, TRANS: tl.constexpr, MIRROR: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                  KC: tl.constexpr, PACKED: tl.constexpr):
    """K2 v2: one program per upper-triangle tile (1-D triangular grid, pid -> (ti <= tj) by the quadratic formula,
    as in NVIDIA Emerging-Optimizers triton_kernels/syrk.py), tiles loaded along their contiguous dimension and
    transposed in registers. Same KC-wide partial sums as K2."""
    pid = tl.program_id(0)
    m = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) / 2.0).to(tl.int32)
    n = pid - m * (m + 1) // 2
    if n > m:
        m += 1
        n = pid - m * (m + 1) // 2
    ri = n * BM + tl.arange(0, BM)
    rj = m * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for kc in range(0, K, KC):
        part = tl.zeros((BM, BM), dtype=tl.float32)
        for k0 in range(kc, kc + KC, BK):
            kk = k0 + rk
            if TRANS:  # A is K x N, contiguous along n: load (BK, BM) tiles
                xt = tl.load(A + kk[:, None] * s_k + ri[None, :] * s_n, mask=(kk[:, None] < K) & (ri[None, :] < N), other=0.0)
                y = tl.load(A + kk[:, None] * s_k + rj[None, :] * s_n, mask=(kk[:, None] < K) & (rj[None, :] < N), other=0.0)
                part = tl.dot(tl.trans(xt), y, part, input_precision="ieee")
            else:  # A is N x K, contiguous along k: load (BM, BK) tiles
                x = tl.load(A + ri[:, None] * s_n + kk[None, :] * s_k, mask=(ri[:, None] < N) & (kk[None, :] < K), other=0.0)
                yt = tl.load(A + rj[:, None] * s_n + kk[None, :] * s_k, mask=(rj[:, None] < N) & (kk[None, :] < K), other=0.0)
                part = tl.dot(x, tl.trans(yt), part, input_precision="ieee")
        acc += part
    if PACKED:  # row-major packed upper triangle: (i, j), i <= j, at i N - i (i - 1) / 2 + j - i
        off = ri[:, None] * N - ri[:, None] * (ri[:, None] - 1) // 2 + rj[None, :] - ri[:, None]
        tl.store(C + off, acc, mask=(ri[:, None] <= rj[None, :]) & (rj[None, :] < N))
    else:
        tl.store(C + ri[:, None] * N + rj[None, :], acc, mask=(ri[:, None] < N) & (rj[None, :] < N))
        if MIRROR:
            if n != m:
                tl.store(C + rj[:, None] * N + ri[None, :], tl.trans(acc), mask=(rj[:, None] < N) & (ri[None, :] < N))


@triton.jit
def _unpack_kernel(P, C, N, BM: tl.constexpr):
    """Full symmetric N x N matrix from its row-major packed upper triangle."""
    ri = tl.program_id(0) * BM + tl.arange(0, BM)
    rj = tl.program_id(1) * BM + tl.arange(0, BM)
    i = tl.minimum(ri[:, None], rj[None, :])
    j = tl.maximum(ri[:, None], rj[None, :])
    mask = (ri[:, None] < N) & (rj[None, :] < N)
    v = tl.load(P + i * N - i * (i - 1) // 2 + j - i, mask=mask)
    tl.store(C + ri[:, None] * N + rj[None, :], v, mask=mask)


def unpack(Pk, n):
    C = torch.empty(n, n, device=Pk.device, dtype=Pk.dtype)
    _unpack_kernel[(triton.cdiv(n, 64), triton.cdiv(n, 64))](Pk, C, n, BM=64)
    return C


def syrk2(A, trans=False, mirror=False):
    """K2 v2 (same contract as syrk). X X^T is computed as Y^T Y with Y = A^T copied contiguous: the X^T X kernel is
    the fast one on both layouts (results/k2v2), and the copy is one O(n k) memory pass."""
    assert A.dim() == 2 and A.dtype == torch.float32
    if not trans:
        A, trans = A.T.contiguous(), True
    if trans:
        K, N = A.shape
        s_k, s_n = A.stride()
    else:
        N, K = A.shape
        s_n, s_k = A.stride()
    C = torch.empty(N, N, device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BM"]) * (triton.cdiv(N, meta["BM"]) + 1) // 2,)
    _syrk2_kernel[grid](A, C, N, K, s_n, s_k, TRANS=trans, MIRROR=mirror, KC=128, PACKED=False)
    return C


def syrk(A, trans=False, mirror=False):
    """trans=False: A @ A.T (A is n x k); trans=True: A.T @ A (A is k x n). Only the upper triangle is written
    unless mirror=True. A may have any strides."""
    assert A.dim() == 2 and A.dtype == torch.float32
    if trans:
        K, N = A.shape
        s_k, s_n = A.stride()
    else:
        N, K = A.shape
        s_n, s_k = A.stride()
    C = torch.empty(N, N, device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BM"]), triton.cdiv(N, meta["BM"]))
    _syrk_kernel[grid](A, C, N, K, s_n, s_k, TRANS=trans, MIRROR=mirror, KC=128)
    return C


def syrk_blocked_torch(A, trans=False, block=512):
    """SYRK in plain PyTorch: one torch.matmul per block row on the upper block pairs (upper triangle written)."""
    X = A.T if trans else A
    N = X.shape[0]
    C = torch.empty(N, N, device=A.device, dtype=A.dtype)
    for i in range(0, N, block):
        C[i:i + block, i:] = X[i:i + block] @ X[i:].T
    return C


def syrk_blocked_mirror(A, trans=False, block=512):
    """syrk_blocked_torch that also mirrors each block row below the diagonal: a full symmetric C from cuBLAS GEMMs
    on the upper block pairs only (half the FLOPs of A @ A.T, up to the diagonal blocks)."""
    X = A.T if trans else A
    N = X.shape[0]
    C = torch.empty(N, N, device=A.device, dtype=A.dtype)
    for i in range(0, N, block):
        blk = X[i:i + block] @ X[i:].T
        C[i:i + block, i:] = blk
        C[i + block:, i:i + block] = blk[:, block:].T
    return C
