#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err__));                              \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

template <int BLOCK_SIZE>
__global__ void row_sum_kernel(const float* x, float* y, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  float local = 0.0f;
  for (int col = tid; col < cols; col += BLOCK_SIZE) {
    local += x[row * cols + col];
  }

  __shared__ float smem[BLOCK_SIZE];
  smem[tid] = local;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[row] = smem[0];
  }
}

template <int BLOCK_SIZE>
__global__ void row_max_kernel(const float* x, float* y, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  float local = -INFINITY;
  for (int col = tid; col < cols; col += BLOCK_SIZE) {
    local = fmaxf(local, x[row * cols + col]);
  }

  __shared__ float smem[BLOCK_SIZE];
  smem[tid] = local;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[row] = smem[0];
  }
}

static bool check_sum(const std::vector<float>& x, const std::vector<float>& y,
                      int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float ref = 0.0f;
    for (int c = 0; c < cols; ++c) {
      ref += x[r * cols + c];
    }
    if (std::fabs(ref - y[r]) > 1e-3f) {
      std::printf("sum mismatch row %d: got %.8f expected %.8f\n", r, y[r],
                  ref);
      return false;
    }
  }
  return true;
}

static bool check_max(const std::vector<float>& x, const std::vector<float>& y,
                      int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float ref = -INFINITY;
    for (int c = 0; c < cols; ++c) {
      ref = std::fmax(ref, x[r * cols + c]);
    }
    if (std::fabs(ref - y[r]) > 1e-5f) {
      std::printf("max mismatch row %d: got %.8f expected %.8f\n", r, y[r],
                  ref);
      return false;
    }
  }
  return true;
}

template <typename LaunchFn>
static float time_kernel(LaunchFn launch, int repeat) {
  for (int i = 0; i < 10; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / repeat;
}

int main() {
  constexpr int BLOCK_SIZE = 256;
  const int rows = 4096;
  const int cols = 1024;
  const int repeat = 100;
  const size_t count = static_cast<size_t>(rows) * cols;

  std::vector<float> h_x(count);
  std::vector<float> h_y(rows);
  for (size_t i = 0; i < count; ++i) {
    h_x[i] = std::sin(static_cast<float>(i) * 0.001f);
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  float sum_ms = time_kernel(
      [&]() { row_sum_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE>>>(d_x, d_y, rows, cols); },
      repeat);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool sum_ok = check_sum(h_x, h_y, rows, cols);

  float max_ms = time_kernel(
      [&]() { row_max_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE>>>(d_x, d_y, rows, cols); },
      repeat);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool max_ok = check_max(h_x, h_y, rows, cols);

  double read_bytes = static_cast<double>(count) * sizeof(float);
  std::printf("rows=%d cols=%d block=%d\n", rows, cols, BLOCK_SIZE);
  std::printf("row_sum_ms=%.4f effective_read_GB/s=%.2f correct=%s\n", sum_ms,
              read_bytes / (sum_ms * 1e-3) / 1e9, sum_ok ? "yes" : "no");
  std::printf("row_max_ms=%.4f effective_read_GB/s=%.2f correct=%s\n", max_ms,
              read_bytes / (max_ms * 1e-3) / 1e9, max_ok ? "yes" : "no");

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}

