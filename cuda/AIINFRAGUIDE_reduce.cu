#include <stdio.h>

// 版本 V0：朴素并行规约
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存
    smem[tid] = gid < n? input[gid]: 0.0f;
    __syncthreads();

    // 树形规约：步长从 1 开始逐步翻倍
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 每个 Block 的结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }

}

// V1: strided index 方式，减少 Warp Divergence
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = gid < n? input[gid]: 0.0f;
    __syncthreads();

    // 步长从 1 开始逐步翻倍，但用 strided index 映射活跃线程
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = threadIdx.x * 2 * s;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// 版本 V2：解决 Bank Conflict
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = gid < n? input[gid]: 0.0f;
    __syncthreads();

    // 步长从 blockDim.x/2 开始，每轮减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// V3: 每线程处理 2 个元素，减少空闲线程
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每个线程加载 2 个相距 blockDim.x 的元素并求和
    float val = 0.0f;
    if (gid < n) val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
    
}

// 辅助函数：展开最后 32 个线程的规约
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}
// V4: 展开最后一个 Warp（在 V3 基础上）
__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每线程处理 2 个元素（继承自 V3）
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();
    // 规约循环仅执行到 step > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockDim.x] = smem[0];
    }
}

// V5: 完全循环展开（模板参数）
template <int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    // 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
    if (BLOCK_SIZE >= 512) {if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads();}
    if (BLOCK_SIZE >= 256) {if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads();}
    if (BLOCK_SIZE >= 128) {if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads();}

    // 最后 Warp 内展开
    if (tid < 32) {
        volatile float* vsmen = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }
    if (tid == 0) output[blockIdx.x] = smem[0];
}
// 调用时需要根据 Block 大小选择对应的模板实例化：
// switch (block_size) {
//     case 512: reduce_v5<512><<<grid, 512, 512*sizeof(float)>>>(d_in, d_out, n); break;
//     case 256: reduce_v5<256><<<grid, 256, 256*sizeof(float)>>>(d_in, d_out, n); break;
//     case 128: reduce_v5<128><<<grid, 128, 128*sizeof(float)>>>(d_in, d_out, n); break;
// }

// Warp 内规约辅助函数
__device__ float warpReduceSum(float val) {
    // 每次将右半边的值加到左半边
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
// V6: Warp Shuffle + 两级规约
__global__ void reduce_v6(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32; // 线程在 Warp 内的编号（0~31）
    int wid = tid / 32; // 该线程属于哪个 Warp

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：Warp 内规约
    val = warpReduceSum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32];  // 最多 32 个 Warp（1024/32）
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    // 第二级：Warp 间规约（用 Warp 0 处理）
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps)? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    if (tid == 0) output[blockIdx.x] = val;
}

// V7: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
__global__ void reduce_v7(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    // float4 加载：每线程每次处理 4 个 float
    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = n / 4; // float4的元素数量
    float val = 0.0f;

    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n4; idx += gridDim.x * blockDim.x) {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 处理 n 不是 4 的倍数时的尾部元素
    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid; idx < n; idx += gridDim.x * blockDim.x) {
        val += input[idx];
    }

    // Warp 内规约
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    if (tid == 0) output[blockIdx.x] = val;
}
// 调用方式
// 固定 Grid 大小为 SM 数 × 4（最大化 GPU 利用率）
// int num_sms;
// cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
// int grid_size  = num_sms * 4;   // 432 for A100
// int block_size = 256;

// reduce_v7<<<grid_size, block_size>>>(d_input, d_partial, n);

int main() {
    const float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    constexpr int n = sizeof(input) / sizeof(input[0]);
    constexpr int threads = 4;
    // constexpr int blocks = (n + threads - 1) / threads;
    constexpr int elements_per_block = 2 * threads;  // V3；V0–V2 用 threads
    constexpr int blocks  = (n + elements_per_block - 1) / elements_per_block;

    // const float expected[blocks] = {10, 26, 19};
    const float expected[blocks] = {36.0f, 19.0f};
    float partial_sums[blocks] = {};

    auto check_cuda = [](cudaError_t error, const char* operation) {
        if (error != cudaSuccess) {
            fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
            std::exit(EXIT_FAILURE);
        }
    };

    float* d_input = nullptr;
    float* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, sizeof(input)), "allocate input");
    check_cuda(cudaMalloc(&d_output, sizeof(partial_sums)), "allocate output");

    check_cuda(
        cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice),
        "copy input to GPU"
    );

    // reduce_v0<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);
    // reduce_v1<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);
    // reduce_v2<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);
    reduce_v3<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);
    check_cuda(cudaGetLastError(), "launch reduce_v0");
    check_cuda(cudaDeviceSynchronize(), "execute reduce_v0");

    check_cuda(
        cudaMemcpy(partial_sums, d_output, sizeof(partial_sums), cudaMemcpyDeviceToHost),
        "copy partial sums to CPU"
    );

    float total = 0.0f;
    bool passed = true;
    for (int block = 0; block < blocks; ++block) {
        printf("block %d: %.0f (expected %.0f)\n",
               block, partial_sums[block], expected[block]);
        passed = passed && (partial_sums[block] == expected[block]);
        total += partial_sums[block];
    }
    printf("CPU sum of partial sums: %.0f\n", total);

    check_cuda(cudaFree(d_input), "free input");
    check_cuda(cudaFree(d_output), "free output");

    printf("%s\n", passed ? "PASS" : "FAIL");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}