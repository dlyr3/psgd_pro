// cuBLAS SYRK and SYMM on PyTorch's cuBLAS handle and stream, for row-major fp32 tensors, TF32 off.
// A row-major r x c matrix is a column-major c x r matrix (its transpose) with leading dimension c; the row-major
// upper triangle is the column-major lower triangle.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

namespace {
struct DefaultMath {  // force exact fp32 (no TF32) for the call, restore the handle's mode afterwards
  cublasHandle_t h;
  cublasMath_t old;
  explicit DefaultMath(cublasHandle_t h) : h(h) {
    cublasGetMathMode(h, &old);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
  }
  ~DefaultMath() { cublasSetMathMode(h, old); }
};
}  // namespace

// trans=false: C = A A^T (A is n x k); trans=true: C = A^T A (A is k x n). Writes the upper triangle of C only.
torch::Tensor syrk(torch::Tensor A, bool trans) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32 && A.dim() == 2, "A: fp32 CUDA matrix");
  A = A.contiguous();
  const int64_t n = trans ? A.size(1) : A.size(0), k = trans ? A.size(0) : A.size(1);
  auto C = torch::empty({n, n}, A.options());
  auto h = at::cuda::getCurrentCUDABlasHandle();
  DefaultMath dm(h);
  const float alpha = 1.f, beta = 0.f;
  // column-major view M = A^T: trans=false -> M is k x n and C = M^T M (op T); trans=true -> M is n x k, C = M M^T.
  TORCH_CUDABLAS_CHECK(cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, trans ? CUBLAS_OP_N : CUBLAS_OP_T, n, k, &alpha,
                                   A.data_ptr<float>(), trans ? n : k, &beta, C.data_ptr<float>(), n));
  return C;
}

// C = S @ B with S (n x n) symmetric, read from its upper triangle only; B (n x p).
torch::Tensor symm(torch::Tensor S, torch::Tensor B) {
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kFloat32 && S.is_contiguous() && S.size(0) == S.size(1), "S");
  B = B.contiguous();
  const int64_t n = S.size(0), p = B.size(1);
  TORCH_CHECK(B.size(0) == n, "shapes");
  auto C = torch::empty({n, p}, B.options());
  auto h = at::cuda::getCurrentCUDABlasHandle();
  DefaultMath dm(h);
  const float alpha = 1.f, beta = 0.f;
  // column-major: C^T (p x n) = B^T (p x n) S, i.e. side right; S's row-major upper = column-major lower.
  TORCH_CUDABLAS_CHECK(cublasSsymm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, p, n, &alpha, S.data_ptr<float>(), n,
                                   B.data_ptr<float>(), p, &beta, C.data_ptr<float>(), p));
  return C;
}

// C = B @ S with S (n x n) symmetric, read from its upper triangle only; B (m x n).
torch::Tensor symm_right(torch::Tensor B, torch::Tensor S) {
  TORCH_CHECK(S.is_cuda() && S.dtype() == torch::kFloat32 && S.is_contiguous() && S.size(0) == S.size(1), "S");
  B = B.contiguous();
  const int64_t m = B.size(0), n = S.size(0);
  TORCH_CHECK(B.size(1) == n, "shapes");
  auto C = torch::empty({m, n}, B.options());
  auto h = at::cuda::getCurrentCUDABlasHandle();
  DefaultMath dm(h);
  const float alpha = 1.f, beta = 0.f;
  // column-major: C^T (n x m) = S B^T, side left; S's row-major upper = column-major lower.
  TORCH_CUDABLAS_CHECK(cublasSsymm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n, m, &alpha, S.data_ptr<float>(), n,
                                   B.data_ptr<float>(), n, &beta, C.data_ptr<float>(), n));
  return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("symm_right", &symm_right, "cublasSsymm, B @ S with S from its upper triangle (row-major)");
  m.def("syrk", &syrk, "cublasSsyrk, upper triangle (row-major)");
  m.def("symm", &symm, "cublasSsymm, S from its upper triangle (row-major)");
}
