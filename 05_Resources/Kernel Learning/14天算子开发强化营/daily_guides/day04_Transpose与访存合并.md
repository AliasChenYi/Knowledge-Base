# Day 04：Transpose 与访存合并

![Transpose 与访存合并](../assets/day04_transpose_coalescing.svg)

## 今天的核心结论

矩阵转置是最好的内存访问入门案例之一。它几乎不做计算，只是搬数据，但 naive 写法会很慢。

原因不是数学难，而是：

```text
相邻线程访问的地址不连续，内存事务效率很差。
```

今天你要第一次真正感受到：**访问模式本身就是性能。**

## 今天你要完成什么

最低 2 小时版：

1. 精读 NVIDIA transpose 官方博客。
2. 写出 copy、naive transpose 的地址映射。
3. 能解释为什么 naive transpose 慢。

完整版 4-6 小时：

1. 实现 copy kernel。
2. 实现 naive transpose。
3. 实现 tiled transpose。
4. 记录三者带宽。

## 必读资料与读法

1. [NVIDIA Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)  
   可靠性：B，NVIDIA 官方博客。今天主读。
2. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)  
   可靠性：A。看 coalesced access 和 shared memory。

读博客时按这个顺序：

1. copy kernel。
2. naive transpose。
3. coalesced transpose。
4. shared memory tile。
5. bank conflict 和 padding。

## 概念讲解：row-major 地址

假设矩阵 `A[height][width]` 按 row-major 存储：

```text
address(A[row][col]) = base + (row * width + col)
```

如果相邻线程负责相邻 `col`：

```text
thread 0 -> A[row][0]
thread 1 -> A[row][1]
thread 2 -> A[row][2]
```

它们访问连续地址，通常更容易 coalesce。

如果相邻线程负责相邻 `row`，固定 col：

```text
thread 0 -> A[0][col]
thread 1 -> A[1][col]
thread 2 -> A[2][col]
```

地址间隔是 `width`，访问不连续。

## 三个 kernel 的意义

### 版本 1：copy

```text
B[y][x] = A[y][x]
```

读连续，写连续。它接近纯内存带宽上限，是一个 baseline。

### 版本 2：naive transpose

```text
B[x][y] = A[y][x]
```

通常读连续，写不连续。写端性能会拖后腿。

### 版本 3：tiled transpose

核心想法：

```text
先把 A 的一个 tile 连续读进 shared memory。
再从 shared memory 转置后，连续写到 B。
```

shared memory 在这里扮演“中转站”，帮助 global memory 两边都尽量连续访问。

## 代码骨架：naive transpose

```cpp
__global__ void transpose_naive(
    const float* in,
    float* out,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}
```

看懂这行：

```cpp
out[x * height + y] = in[y * width + x];
```

`in` 按行读，多数情况下连续；`out` 按列写，相邻线程写的地址跨度大。

## 代码骨架：tiled transpose

```cpp
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_tiled(
    const float* in,
    float* out,
    int width,
    int height
) {
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
```

解释：

- `tile[TILE_DIM][TILE_DIM + 1]` 里的 `+1` 常用于缓解 bank conflict。
- 第一次循环从 global memory 连续读。
- `__syncthreads()` 确保 tile 写完。
- 第二次循环转置写出，尽量让 global memory 写连续。

## 实验表

| 矩阵大小 | copy GB/s | naive transpose GB/s | tiled transpose GB/s | 观察 |
|---|---:|---:|---:|---|
| 1024 x 1024 | | | | |
| 2048 x 2048 | | | | |
| 4096 x 4096 | | | | |

## 常见错误排查

### tiled transpose 结果错

可能原因：

- 第二阶段 x/y 交换错。
- 输出矩阵 stride 用错。
- 少了 `__syncthreads()`。
- 边界条件对 width/height 判断错。

### tiled 版本没变快

可能原因：

- 矩阵太小，launch overhead 影响大。
- tile 参数不合适。
- 没处理 bank conflict。
- 计时包含了 host-device copy。

## 自测题

1. naive transpose 为什么读连续但写不连续？
2. shared memory 在 transpose 中解决什么问题？
3. `__syncthreads()` 为什么必须？
4. `TILE_DIM + 1` 可能有什么用？
5. copy kernel 为什么是一个好 baseline？

参考答案：

1. row-major 下相邻线程读同一行相邻列，但写到输出的不同列/行映射导致跨度大。
2. 改变访问模式，让 global memory 两端尽量连续。
3. 防止有线程还没写完 tile，其他线程就开始读。
4. 减少 shared memory bank conflict。
5. 它接近纯连续读写带宽，用来比较 transpose 的损失。

## 今天的验收标准

你能清楚说出：

```text
transpose 的优化不是减少计算，而是把不连续的 global memory 访问变成更连续的访问。shared memory 作为中转 tile，让读写两端都更接近 coalesced。
```

