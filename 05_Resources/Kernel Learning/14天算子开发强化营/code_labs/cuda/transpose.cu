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

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void copy_kernel(const float* in, float* out, int width,
                            int height) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && y + j < height) {
      out[(y + j) * width + x] = in[(y + j) * width + x];
    }
  }
}

__global__ void transpose_naive_kernel(const float* in, float* out, int width,
                                       int height) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && y + j < height) {
      out[x * height + (y + j)] = in[(y + j) * width + x];
    }
  }
}

__global__ void transpose_tiled_kernel(const float* in, float* out, int width,
                                       int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && y + j < height) {
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < height && y + j < width) {
      out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

using KernelFn = void (*)(const float*, float*, int, int);

static float time_kernel(KernelFn kernel, const float* d_in, float* d_out,
                         int width, int height, dim3 grid, dim3 block,
                         int repeat) {
  for (int i = 0; i < 10; ++i) {
    kernel<<<grid, block>>>(d_in, d_out, width, height);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    kernel<<<grid, block>>>(d_in, d_out, width, height);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / repeat;
}

static bool check_transpose(const std::vector<float>& in,
                            const std::vector<float>& out, int width,
                            int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float ref = in[y * width + x];
      float got = out[x * height + y];
      if (std::fabs(ref - got) > 1e-5f) {
        std::printf("Mismatch at input(%d,%d): got %.8f expected %.8f\n", y, x,
                    got, ref);
        return false;
      }
    }
  }
  return true;
}

int main() {
  const int width = 4096;
  const int height = 4096;
  const int repeat = 100;
  const size_t count = static_cast<size_t>(width) * height;

  std::vector<float> h_in(count);
  std::vector<float> h_out(count);
  for (size_t i = 0; i < count; ++i) {
    h_in[i] = static_cast<float>(i % 1024);
  }

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
            (height + TILE_DIM - 1) / TILE_DIM);

  float copy_ms =
      time_kernel(copy_kernel, d_in, d_out, width, height, grid, block, repeat);
  float naive_ms = time_kernel(transpose_naive_kernel, d_in, d_out, width,
                               height, grid, block, repeat);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool naive_ok = check_transpose(h_in, h_out, width, height);

  float tiled_ms = time_kernel(transpose_tiled_kernel, d_in, d_out, width,
                               height, grid, block, repeat);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool tiled_ok = check_transpose(h_in, h_out, width, height);

  double bytes = static_cast<double>(count) * 2.0 * sizeof(float);
  std::printf("copy_ms=%.4f GB/s=%.2f\n", copy_ms,
              bytes / (copy_ms * 1e-3) / 1e9);
  std::printf("naive_ms=%.4f GB/s=%.2f correct=%s\n", naive_ms,
              bytes / (naive_ms * 1e-3) / 1e9, naive_ok ? "yes" : "no");
  std::printf("tiled_ms=%.4f GB/s=%.2f correct=%s\n", tiled_ms,
              bytes / (tiled_ms * 1e-3) / 1e9, tiled_ok ? "yes" : "no");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}

