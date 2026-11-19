#include <stdio.h>

// V0: 朴素实现，一个线程处理一行
__global__ void softmax_v0(float* input, float* output, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row > M) return;

    float* x = input + row * N; // 每行对应的起点
    float* y = output + row * N;

    // Pass 1: 求最大值
    float max_val = -INFINITY;
    for (int i = 0; i < N; i ++) {
        max_val = fmax(max_val, x[i]);
    }

    // Pass 2: 指数求和
    float sum = 0.0f;
    for (int i = 0; i < N; i ++) {
        sum = expf(x[i] - max_val);
    }

    // Pass 3: 归一化
    float inv_sum = 1.0 / sum;
    for (int i = 0; i < N; i ++) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// V1: 一个 Block 处理一行，Block 内并行规约
__global__ void softmax_v1(float* input, float* output, int M, int N) {
    extern __shared__ float smem[]; // 用于规约
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    // Pass 1: 并行求最大值
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmax(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();

    // Shared Memory 规约求全局最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmax(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Pass 2: 并行求指数和
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    smem[tid] = sum;
    __syncthreads();

    // Shared Memory 规约求总和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Pass 3: 归一化写出
    float inv_sum = 1.0 / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// V2: Warp Shuffle 两级规约
// Warp 内规约：Max
__device__ float warpReduceMax(float val) {
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
// Warp 内规约：Sum
__device__ float warpReduceSum(float val) {
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
// 两级规约：Block 内 Max
__device__ float blockReduceMaxShuffle(float val) {
    __shared__ float warp_max[32];

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);
    if (lane == 0) warp_max[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    val = lane < num_warps? warp_max[lane]: -INFINITY;

    if (wid == 0) val = warpReduceMax(val);
    // 广播结果给所有线程
    __shared__ float block_result;
    if (threadIdx.x == 0) block_result = val;
    __syncthreads();
    return block_result;
}
// 两级规约：Block 内 Sum
__device__ float blockReduceSumShuffle(float val) {
    __shared__ float warp_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) warp_sum[wid] = val;
    __syncthreads();
    int num_warps = blockDim.x / 32;
    val = lane < num_warps? warp_sum[lane]: 0.0f;

    if (wid == 0) val = warpReduceSum(val);
    // 广播结果给所有线程
    __shared__ float block_result;
    if (threadIdx.x == 0) block_result = val;
    __syncthreads();
    return block_result;
}
// V2: Warp Shuffle 两级规约
__global__ void softmax_v2(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    // Pass 1: 求行最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmax(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 求指数和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);

    // Pass 3: 归一化
    float inv_sum = 1.0 / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// V3: float4 向量化加载
__global__ void softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 向量化指针
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4; // float4 元素数

    // Pass 1: 向量化求最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmax(local_max, fmax(fmax(data.x, data.y), fmax(data.z, data.w)));
    }
    // 处理尾部元素（N 不是 4 的倍数时）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_max = fmax(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 向量化求指数和
    float local_sum = 0.0;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_sum += expf(data.x - max_val) + expf(data.y - max_val) + expf(data.z - max_val) + expf(data.w - max_val);
    }
    // 处理尾部元素（N 不是 4 的倍数时）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);

    // Pass 3: 向量化归一化
    float inv_sum = 1.0 / sum;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 result;
        result.x = expf(data.x - max_val) * inv_sum;
        result.y = expf(data.y - max_val) * inv_sum;
        result.z = expf(data.z - max_val) * inv_sum;
        result.w = expf(data.w - max_val) * inv_sum;
        y4[i] = result;
    }
    // 处理尾部元素（N 不是 4 的倍数时）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// V4: 两遍 Kernel（Pass 1 求 max，Pass 2 融合 exp-sum + normalize）
// 需要额外 Shared Memory 暂存 exp 值
__global__ void softmax_v4(float* input, float* output, int M, int N) {
    extern __shared__ float smem[];  // 大小 = N（暂存 exp 值）
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 向量化指针
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4; // float4 元素数

    // Pass 1: 向量化求最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmax(local_max, fmax(fmax(data.x, data.y), fmax(data.z, data.w)));
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_max = fmax(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 计算 exp 并暂存，同时累加求 sum
    float local_sum  = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        smem[i] = exp_val;
        local_sum += exp_val;
    }
    float sum = blockReduceSumShuffle(local_sum);

    // 直接从 Shared Memory 读取 exp 值并归一化写出
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = smem[i] * inv_sum;
    }
}

int main() {
    return 0;
}