"""K1: triangle-reading V @ A (kernels/k1.cu) and the norm lower bounds of psgd_pro built on it."""
import os
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
_ext = load(name="k1_ext", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
tri_vmm, tri_rownorm2 = _ext.tri_vmm, _ext.tri_rownorm2


def sym_row(A, j, sign):
    """Row j of the symmetric (sign=1) or skew-symmetric (sign=-1) matrix held in the upper triangle of A."""
    j = j.view(1)
    row, col = A.index_select(0, j)[0], A.index_select(1, j)[:, 0]
    return torch.where(torch.arange(A.shape[0], device=A.device) >= j, row, sign * col)


def norm_lower_bound_tri(A, V, sign):
    """psgd_pro.norm_lower_bound_spd (sign=1) / _skh (sign=-1) with the random start V passed in. Reads only the
    upper triangle of A (of A^T if A is column-major, as R = Q.H - Q is): one pass for the row norms and max|A|,
    four for V @ A; A / normalizing_factor is folded into the 32 x n products instead of being written out."""
    A, flip = (A, 1) if A.is_contiguous() else (A.T, sign)  # column-major storage: A^T = sign * A, read A^T
    sn = torch.finfo(A.dtype).smallest_normal
    rn2, amax = tri_rownorm2(A)
    nf = (A.diagonal().amax() if sign > 0 else amax) + sn
    Aj = (flip * sym_row(A, torch.argmax(rn2), sign) / nf).unsqueeze(0)
    V = Aj + torch.sgn(torch.sum(Aj * V, dim=1, keepdim=True)) * V
    for _ in range(2):
        V = flip * tri_vmm(V.contiguous(), A, sign) / nf
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + sn
        V = flip * tri_vmm(V.contiguous(), A, sign) / nf
    return nf * torch.amax(torch.linalg.vector_norm(V, dim=1))


tri_vmm_diag = _ext.tri_vmm_diag
tri_vmm_packed = _ext.tri_vmm_packed


def pack_upper(A):
    """Row-major packed upper triangle (diagonal included), n(n+1)/2 values."""
    iu = torch.triu_indices(A.shape[0], A.shape[0], device=A.device)
    return A[iu[0], iu[1]].contiguous()
