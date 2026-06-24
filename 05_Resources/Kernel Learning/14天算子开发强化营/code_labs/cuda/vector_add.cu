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

__global__ void vector_add_kernel(const float* a, const float* b, float* c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

static void fill_input(std::vector<float>& x, float scale) {
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = std::sin(i * 0.001f) * scale;
  }
}

static bool check_result(const std::vector<float>& a,
                         const std::vector<float>& b,
                         const std::vector<float>& c) {
  for (int i = 0; i < static_cast<int>(c.size()); ++i) {
    float ref = a[i] + b[i];
    if (std::fabs(ref - c[i]) > 1e-5f) {
      std::printf("Mismatch at %d: got %.8f, expected %.8f\n", i, c[i], ref);
      return false;
    }
  }
  return true;
}

static float run_benchmark(const float* d_a, const float* d_b, float* d_c,
                           int n, int block_size, int repeat) {
  int grid_size = (n + block_size - 1) / block_size;

  for (int i = 0; i < 10; ++i) {
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
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
  const int n = 1 << 24;
  const int repeat = 100;

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n);
  fill_input(h_a, 1.0f);
  fill_input(h_b, 2.0f);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int block_sizes[] = {128, 256, 512};
  for (int block_size : block_sizes) {
    float avg_ms = run_benchmark(d_a, d_b, d_c, n, block_size, repeat);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok = check_result(h_a, h_b, h_c);

    double bytes = static_cast<double>(n) * 3.0 * sizeof(float);
    double gbps = bytes / (avg_ms * 1e-3) / 1e9;
    std::printf("N=%d block=%d avg_ms=%.4f GB/s=%.2f correct=%s\n", n,
                block_size, avg_ms, gbps, ok ? "yes" : "no");
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}

