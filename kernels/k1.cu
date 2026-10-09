// K1: W = V @ A for a symmetric (sign = +1) or skew-symmetric (sign = -1) n x n fp32 matrix A stored as
// triangle-in-square: only the upper triangle (diagonal included) is ever read, so the lower triangle may hold
// anything, NaN included. V is 32 x n. Also the squared row norms of A and max|A_ij| from the same triangle.
//
// Schedule. A is cut into N = n/64 column blocks. CTA (J, s) owns output block W[:, J]: it needs block column J
// of the full matrix, i.e. one stored tile per partner block P (tile (P, J) if P <= J, else tile (J, P) transposed).
// The rounds follow a round-robin tournament (circle method): in round r every block is paired with exactly one
// other block, so the two CTAs that need the same stored tile (J's and P's) load it in the same round, and the
// second load should hit L2 instead of DRAM. Round M-1 is each block's own diagonal tile. With S > 1, the rounds
// of a block are split over S CTAs (r = s mod S) and the S partial outputs are summed afterwards in a fixed order.
// No atomics: the result is deterministic.
//
// Thread (c, kg) of the 512-thread CTA holds 8 values of a tile, column c and rows 8kg..8kg+7 of the (virtual)
// block A[P, J], and accumulates the 32 outputs W[0..31][64J + c] over those rows; the V values it needs are
// broadcast across its warp from shared memory. The 8 k-groups are reduced through shared memory at the end.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
constexpr int T = 64, R = 32, KG = 8, NT = T * KG;

__device__ __forceinline__ int partner(int J, int rd, int M) {
  if (rd == M - 1) return J;
  if (J == M - 1) return rd;
  if (J == rd) return M - 1;
  int p = (2 * rd - J) % (M - 1);
  return p < 0 ? p + M - 1 : p;
}

// a[i] = Afull[64P + k0 + i][64J + c], reading only the upper triangle of A.
template <int SIGN>
__device__ __forceinline__ void load_vals(const float* __restrict__ A, int n, int J, int P, int c, int k0, float* a) {
  if (P < J) {
    const float* p = A + (size_t)(P * T + k0) * n + J * T + c;
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = __ldg(p + (size_t)i * n);
  } else if (P > J) {
    const float4* p = reinterpret_cast<const float4*>(A + (size_t)(J * T + c) * n + P * T + k0);
    float4 x = __ldg(p), y = __ldg(p + 1);
    a[0] = SIGN * x.x; a[1] = SIGN * x.y; a[2] = SIGN * x.z; a[3] = SIGN * x.w;
    a[4] = SIGN * y.x; a[5] = SIGN * y.y; a[6] = SIGN * y.z; a[7] = SIGN * y.w;
  } else {
    const float* b = A + (size_t)(J * T) * n + J * T;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int k = k0 + i;
      if (k < c) a[i] = __ldg(b + (size_t)k * n + c);
      else if (k > c) a[i] = SIGN * __ldg(b + (size_t)c * n + k);
      else a[i] = SIGN > 0 ? __ldg(b + (size_t)k * n + k) : 0.f;
    }
  }
}


// Packed storage (stretch): the upper triangle row by row, n(n+1)/2 values; row i starts at i*n - i(i-1)/2.
__device__ __forceinline__ size_t poff(int i, int n) { return (size_t)i * n - (size_t)i * (i - 1) / 2; }

template <int SIGN>
__device__ __forceinline__ void load_vals_packed(const float* __restrict__ Ap, int n, int J, int P, int c, int k0, float* a) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = k0 + i;
    if (P < J) {
      const int r = P * T + k, col = J * T + c;
      a[i] = __ldg(Ap + poff(r, n) + (col - r));
    } else if (P > J) {
      const int r = J * T + c, col = P * T + k;
      a[i] = SIGN * __ldg(Ap + poff(r, n) + (col - r));
    } else if (k < c) {
      a[i] = __ldg(Ap + poff(J * T + k, n) + (c - k));
    } else if (k > c) {
      a[i] = SIGN * __ldg(Ap + poff(J * T + c, n) + (k - c));
    } else {
      a[i] = SIGN > 0 ? __ldg(Ap + poff(J * T + k, n)) : 0.f;
    }
  }
}

// NORMS = false: out[s][r][64J + c] = partial (V @ A)[r][64J + c] over the rounds of split s.
// NORMS = true:  out[s][64J + c] = partial sum of squares of row 64J + c; amax_bits = max |A_ij| (as int bits).
// DIAG (diagnostics only): 0 normal, 1 global loads without the FMA work, 2 FMA work without the A loads.
// SCHED (diagnostics only): 0 tournament pairing, 1 natural order P = round (the two loads of a tile are far apart).
template <int SIGN, bool NORMS, int DIAG = 0, int MINB = 1, int SCHED = 0>
__global__ void __launch_bounds__(NT, MINB) tri_kernel(const float* __restrict__ A, const float* __restrict__ V,
                                                 float* __restrict__ out, int* __restrict__ amax_bits,
                                                 int n, int N, int M, int S) {
  const int J = blockIdx.x / S, s = blockIdx.x % S;
  const int tid = threadIdx.x, c = tid % T, kg = tid / T, k0 = kg * 8;
  __shared__ __align__(16) float vs[R * T];
  __shared__ float red[KG * 8 * T];
  constexpr int NA = NORMS ? 1 : R;
  float acc[NA];
#pragma unroll
  for (int r = 0; r < NA; ++r) acc[r] = 0.f;
  float amax = 0.f;

  auto pick = [&](int rd) { return SCHED ? rd : partner(J, rd, M); };
  auto next_round = [&](int rd) {  // first round >= rd of this split whose partner exists
    while (rd < M && pick(rd) >= N) rd += S;
    return rd;
  };
  int rd = next_round(s);
  float a[8], an[8];
  float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f), vn = v4;
  const int vr = (tid * 4) / T, vc = (tid * 4) % T;
  if (rd < M) {
    const int P = pick(rd);
    if (DIAG == 2) {
#pragma unroll
      for (int i = 0; i < 8; ++i) a[i] = an[i] = 1e-3f * (c + i);
    } else load_vals<SIGN>(A, n, J, P, c, k0, a);
    if (!NORMS) v4 = __ldg(reinterpret_cast<const float4*>(V + (size_t)vr * n + P * T + vc));
  }
  while (rd < M) {
    const int nxt = next_round(rd + S);
    if (nxt < M) {  // prefetch the next round while this one computes
      const int Pn = pick(nxt);
      if (DIAG != 2) load_vals<SIGN>(A, n, J, Pn, c, k0, an);
      if (!NORMS) vn = __ldg(reinterpret_cast<const float4*>(V + (size_t)vr * n + Pn * T + vc));
    }
    if (NORMS) {
#pragma unroll
      for (int i = 0; i < 8; ++i) { acc[0] = fmaf(a[i], a[i], acc[0]); amax = fmaxf(amax, fabsf(a[i])); }
    } else if (DIAG == 1) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += a[i] + v4.x;
    } else {
      __syncthreads();
      reinterpret_cast<float4*>(vs)[tid] = v4;
      __syncthreads();
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float4 x = *reinterpret_cast<const float4*>(vs + r * T + k0);
        const float4 y = *reinterpret_cast<const float4*>(vs + r * T + k0 + 4);
        float t = acc[r];
        t = fmaf(x.x, a[0], t); t = fmaf(x.y, a[1], t); t = fmaf(x.z, a[2], t); t = fmaf(x.w, a[3], t);
        t = fmaf(y.x, a[4], t); t = fmaf(y.y, a[5], t); t = fmaf(y.z, a[6], t); t = fmaf(y.w, a[7], t);
        acc[r] = t;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = an[i];
    v4 = vn;
    rd = nxt;
  }

  if (NORMS) {
    red[kg * T + c] = acc[0];
    __syncthreads();
    if (kg == 0) {
      float t = 0.f;
#pragma unroll
      for (int g = 0; g < KG; ++g) t += red[g * T + c];
      out[(size_t)s * n + J * T + c] = t;
    }
    // max |A_ij|: warp then block reduce, then one atomicMax on the non-negative float's bits (order-free)
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    __syncthreads();
    if ((tid & 31) == 0) red[tid / 32] = amax;
    __syncthreads();
    if (tid == 0) {
      float m = 0.f;
      for (int w = 0; w < NT / 32; ++w) m = fmaxf(m, red[w]);
      atomicMax(amax_bits, __float_as_int(m));
    }
  } else {
#pragma unroll
    for (int q = 0; q < R / 8; ++q) {
      __syncthreads();
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) red[(kg * 8 + rr) * T + c] = acc[q * 8 + rr];
      __syncthreads();
      const int rr = tid / T;  // 512 threads = 8 rows x 64 columns
      float t = 0.f;
#pragma unroll
      for (int g = 0; g < KG; ++g) t += red[(g * 8 + rr) * T + c];
      out[((size_t)s * R + q * 8 + rr) * n + J * T + c] = t;
    }
  }
}


// v2 of the V @ A kernel. Same schedule and same global loads as tri_kernel, but the tile and the V slice are
// staged in shared memory and each thread computes an 8 (rows of V) x 4 (columns) block over 8 values of k:
// 32 FMAs per 3 shared loads instead of 8 per 2 (v1 was bound by shared-memory instruction throughput:
// 8.3 TFLOP/s with the A loads removed, results/k1_diag.json). Thread layout: cg = tid % 16 (4 columns each),
// rg = (tid / 16) % 4 (8 rows each), kg = tid / 64 (8 k each); loads use (c = tid % 64, kg) as in v1.
template <int SIGN, bool PACKED = false>
__global__ void __launch_bounds__(NT) tri_vmm_kernel2(const float* __restrict__ A, const float* __restrict__ V,
                                                       float* __restrict__ out, int n, int N, int M, int S) {
  constexpr int AP = T + 4, VP = R + 4;  // padded pitches, 16-byte aligned rows
  const int J = blockIdx.x / S, s = blockIdx.x % S, tid = threadIdx.x;
  const int lc = tid % T, lk0 = (tid / T) * 8;                           // load mapping
  const int cg = tid % 16, rg = (tid / 16) % 4, kg = tid / 64, k0 = kg * 8;  // compute mapping
  const int vr = tid % R, vk = (tid / R) * 4;                            // V slice load mapping (64 = 16 x 4)
  __shared__ __align__(16) float As[T * AP];
  __shared__ __align__(16) float Vt[T * VP];
  float acc[8][4];
#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[r][j] = 0.f;

  auto next_round = [&](int rd) {
    while (rd < M && partner(J, rd, M) >= N) rd += S;
    return rd;
  };
  int rd = next_round(s);
  float a[8], an[8];
  float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f), vn = v4;
  if (rd < M) {
    const int P = partner(J, rd, M);
    if (PACKED) load_vals_packed<SIGN>(A, n, J, P, lc, lk0, a);
    else load_vals<SIGN>(A, n, J, P, lc, lk0, a);
    v4 = __ldg(reinterpret_cast<const float4*>(V + (size_t)vr * n + P * T + vk));
  }
  while (rd < M) {
    const int nxt = next_round(rd + S);
    if (nxt < M) {
      const int Pn = partner(J, nxt, M);
      if (PACKED) load_vals_packed<SIGN>(A, n, J, Pn, lc, lk0, an);
      else load_vals<SIGN>(A, n, J, Pn, lc, lk0, an);
      vn = __ldg(reinterpret_cast<const float4*>(V + (size_t)vr * n + Pn * T + vk));
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) As[(lk0 + i) * AP + lc] = a[i];
    Vt[(vk + 0) * VP + vr] = v4.x; Vt[(vk + 1) * VP + vr] = v4.y;
    Vt[(vk + 2) * VP + vr] = v4.z; Vt[(vk + 3) * VP + vr] = v4.w;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 x = *reinterpret_cast<const float4*>(As + (k0 + i) * AP + cg * 4);
      const float4 v0 = *reinterpret_cast<const float4*>(Vt + (k0 + i) * VP + rg * 8);
      const float4 v1 = *reinterpret_cast<const float4*>(Vt + (k0 + i) * VP + rg * 8 + 4);
      const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        acc[r][0] = fmaf(vv[r], x.x, acc[r][0]);
        acc[r][1] = fmaf(vv[r], x.y, acc[r][1]);
        acc[r][2] = fmaf(vv[r], x.z, acc[r][2]);
        acc[r][3] = fmaf(vv[r], x.w, acc[r][3]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = an[i];
    v4 = vn;
    rd = nxt;
  }
  // reduce over the 8 k-groups, one row group (8 rows) at a time, through As (8 x 8 x 64 floats fit in it)
  float* red = As;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __syncthreads();
    if (rg == q) {
#pragma unroll
      for (int r = 0; r < 8; ++r)
        *reinterpret_cast<float4*>(red + (kg * 8 + r) * T + cg * 4) = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
    __syncthreads();
    const int r = tid / T, c = tid % T;
    float t = 0.f;
#pragma unroll
    for (int g = 0; g < KG; ++g) t += red[(g * 8 + r) * T + c];
    out[((size_t)s * R + q * 8 + r) * n + J * T + c] = t;
  }
}

int pick_splits(int N, int M, int per_sm = 1) {
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return std::max(1, std::min(M, per_sm * sms / N));
}

void check(const torch::Tensor& A) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32 && A.is_contiguous() && A.dim() == 2 &&
              A.size(0) == A.size(1) && A.size(0) % T == 0, "A: contiguous fp32 CUDA n x n with n % 64 == 0");
}
}  // namespace

// W = V @ A_sym, V (32, n) fp32 contiguous; sign = +1 symmetric, -1 skew-symmetric.
torch::Tensor tri_vmm(torch::Tensor V, torch::Tensor A, int64_t sign) {
  check(A);
  const int n = A.size(0), N = n / T, M = N + (N & 1), S = pick_splits(N, M);
  TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kFloat32 && V.is_contiguous() && V.size(0) == R && V.size(1) == n,
              "V: contiguous fp32 (32, n)");
  const c10::cuda::CUDAGuard guard(A.device());
  auto out = torch::empty({S, R, n}, V.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (sign > 0) tri_vmm_kernel2<1><<<N * S, NT, 0, stream>>>(A.data_ptr<float>(), V.data_ptr<float>(), out.data_ptr<float>(), n, N, M, S);
  else tri_vmm_kernel2<-1><<<N * S, NT, 0, stream>>>(A.data_ptr<float>(), V.data_ptr<float>(), out.data_ptr<float>(), n, N, M, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return S == 1 ? out[0] : out.sum(0);
}

// (squared row norms (n,), max |A_ij| (0-d)) of the symmetric / skew-symmetric matrix.
std::vector<torch::Tensor> tri_rownorm2(torch::Tensor A) {
  check(A);
  const int n = A.size(0), N = n / T, M = N + (N & 1), S = pick_splits(N, M);
  const c10::cuda::CUDAGuard guard(A.device());
  auto out = torch::empty({S, n}, A.options());
  auto amax = torch::zeros({}, A.options().dtype(torch::kInt32));
  tri_kernel<1, true><<<N * S, NT, 0, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), nullptr, out.data_ptr<float>(), amax.data_ptr<int>(), n, N, M, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {S == 1 ? out[0] : out.sum(0), amax.view(torch::kFloat32)};
}

// Diagnostic launches of the V @ A kernel (symmetric): diag 0/1/2 as in tri_kernel, minb = CTAs per SM targeted.
torch::Tensor tri_vmm_diag(torch::Tensor V, torch::Tensor A, int64_t diag, int64_t minb) {
  check(A);
  const int n = A.size(0), N = n / T, M = N + (N & 1), S = pick_splits(N, M, minb);
  auto out = torch::empty({S, R, n}, V.options());
  auto st = at::cuda::getCurrentCUDAStream();
  const float* a = A.data_ptr<float>(); const float* v = V.data_ptr<float>(); float* o = out.data_ptr<float>();
  if (diag == 3) {
    tri_vmm_kernel2<1><<<N * S, NT, 0, st>>>(a, v, o, n, N, M, S);
  } else if (diag == 4) {
    tri_kernel<1, false, 1, 1, 1><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
  } else if (diag == 5) {
    tri_kernel<1, false, 0, 1, 1><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
  } else if (minb == 1) {
    if (diag == 0) tri_kernel<1, false, 0, 1><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
    if (diag == 1) tri_kernel<1, false, 1, 1><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
    if (diag == 2) tri_kernel<1, false, 2, 1><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
  } else {
    if (diag == 0) tri_kernel<1, false, 0, 2><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
    if (diag == 1) tri_kernel<1, false, 1, 2><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
    if (diag == 2) tri_kernel<1, false, 2, 2><<<N * S, NT, 0, st>>>(a, v, o, nullptr, n, N, M, S);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return S == 1 ? out[0] : out.sum(0);
}

// W = V @ A_sym with A given packed (n(n+1)/2 values, row-major upper triangle).
torch::Tensor tri_vmm_packed(torch::Tensor V, torch::Tensor Ap, int64_t n, int64_t sign) {
  TORCH_CHECK(Ap.is_cuda() && Ap.dtype() == torch::kFloat32 && Ap.is_contiguous() && Ap.numel() == n * (n + 1) / 2 &&
              n % T == 0, "Ap: packed fp32 upper triangle, n % 64 == 0");
  const int N = n / T, M = N + (N & 1), S = pick_splits(N, M);
  auto out = torch::empty({S, R, n}, V.options());
  auto st = at::cuda::getCurrentCUDAStream();
  if (sign > 0) tri_vmm_kernel2<1, true><<<N * S, NT, 0, st>>>(Ap.data_ptr<float>(), V.data_ptr<float>(), out.data_ptr<float>(), n, N, M, S);
  else tri_vmm_kernel2<-1, true><<<N * S, NT, 0, st>>>(Ap.data_ptr<float>(), V.data_ptr<float>(), out.data_ptr<float>(), n, N, M, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return S == 1 ? out[0] : out.sum(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_vmm_packed", &tri_vmm_packed, "V @ A_sym from packed upper-triangle storage");
  m.def("tri_vmm_diag", &tri_vmm_diag, "diagnostic variants of tri_vmm");
  m.def("tri_vmm", &tri_vmm, "V @ A_sym reading the upper triangle of A");
  m.def("tri_rownorm2", &tri_rownorm2, "squared row norms and max|A| from the upper triangle");
}
