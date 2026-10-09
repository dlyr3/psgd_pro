"""cuBLAS SYRK / SYMM (kernels/blas.cpp) on PyTorch's handle; links the libcublas.so already loaded by torch (same soname)."""
import os
from torch.utils.cpp_extension import load

_ext = load(name="blas_ext", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "blas.cpp")],
            extra_ldflags=["-lcublas"], with_cuda=True, verbose=False)
syrk, symm, symm_right = _ext.syrk, _ext.symm, _ext.symm_right
