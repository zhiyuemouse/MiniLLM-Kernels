#include <stdio.h>

// V0: 单线程 Online Softmax（一个线程处理一行）
__global__ void online_softmax_v0(float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float* x = input + row * N;
    float* y = output + row * N;

    // 第 1 遍：Online 同时计算 max 和 sum
    float m = -INFINITY;
    float d = 0.0f;
    for (int i = 0; i < N; i ++) {
        float x1 = x[i];
        float m_new = fmax(m, x1);
        d = d * expf(m - m_new) + expf(x1 - m_new);
        m = m_new;
    }

    // 第 2 遍：归一化
    float inv_sum = 1.0 / d;
    for (int i = 0; i < N; i ++) {
        y[i] = expf(x[i] - m) * inv_sum;
    }
}

// V1: Block 级并行 + Warp Shuffle 两级合并规约
// Warp 内 Online 合并规约
__device__ void warpReduceOnline(float& m, float& d, const int Warp_Threads = 32) {
    for (unsigned int offset = Warp_Threads / 2; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmax(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}
// V1: Block 级并行 + Warp Shuffle 两级合并规约
__global__ void online_softmax_v1(float* input, float* output, int M, int N, const int Warp_Size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 第 1 遍：每线程 Online 处理一段
    float local_m = -INFINITY;
    float local_d = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmax(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + exp(xi - m_new);
        local_m = m_new;
    }
    // 第一级：Warp 内合并
    warpReduceOnline(local_m, local_d, Warp_Size);

    // Warp 间通过 Shared Memory 交换
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    // 第二级：Warp 0 做最终合并
    int num_warps = blockDim.x / Warp_Size;
    if (wid == 0) {
        local_m = lane < num_warps ? warp_m[lane] : -INFINITY;
        local_d = lane < num_warps ? warp_d[lane] : 0.0f;
        warpReduceOnline(local_m, local_d, Warp_Size);
    }

    // 广播最终结果
    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0 / final_d;

    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = exp(x[i] - row_max) * inv_sum;
    }
}

// V2: float4 向量化加载 + Online Softmax
__global__ void online_softmax_v2(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float* x = input  + row * N;
    float* y = output + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;


    // 第 1 遍：向量化 Online 递推
    float local_m = -INFINITY;
    float local_d = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        // 对 4 个元素依次更新 (m, d)
        float vals[4] = {data.x, data.y, data.z, data.w};
        for (int k = 0; k < 4; k ++) {
            float m_new = fmax(local_m, vals[k]);
            local_d = local_d * expf(local_m - m_new) + exp(vals[k] - m_new);
            local_m = m_new;
        }
    }
    // 尾部处理
    for (int i = N4 * 4 + tid; i < N; i ++) {
        float xi = x[i];
        float m_new = fmax(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }
    // 两级 Warp Shuffle 合并（与 V1 相同）
    warpReduceOnline(local_m, local_d);

    // Warp 间通过 Shared Memory 交换
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    // 第二级：Warp 0 做最终合并
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = lane < num_warps ? warp_m[lane] : -INFINITY;
        local_d = lane < num_warps ? warp_d[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    // 广播最终结果
    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0 / final_d;

    // 第 2 遍：向量化归一化
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 result;
        result.x = (data.x - row_max) * inv_sum;
        result.y = (data.y - row_max) * inv_sum;
        result.z = (data.z - row_max) * inv_sum;
        result.w = (data.w - row_max) * inv_sum;
        y4[i] = result;
    }

    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = exp(x[i] - row_max) * inv_sum;
    }
}

// V3: 寄存器缓存实现 One-Pass Softmax
// 适用于 N / blockDim.x 较小的场景（每线程处理元素数不超过 MAX_ELEMS）
#define MAX_ELEMS_PER_THREAD 32

__global__ void online_softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float* x = input + row * N;
    float* y = output + row * N;

    // 寄存器数组缓存输入数据
    float reg_cache[MAX_ELEMS_PER_THREAD];
    // 第 1 遍（也是唯一一遍全局读取）：读数据 + Online 递推
    float local_m = -INFINITY;
    float local_d = 0.0f;
    int count = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        reg_cache[count] = xi;
        count ++;
        float m_new = fmax(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }
    // 两级 Warp Shuffle 合并
    warpReduceOnline(local_m, local_d);

    __shared__ float warm_m[32];
    __shared__ float warm_d[32];
    if (lane == 0) {
        warm_m[wid] = local_m;
        warm_d[wid] = local_d;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = lane < num_warps ? warm_m[lane] : -INFINITY;
        local_d = lane < num_warps ? warm_d[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0 / final_d;

    // 从寄存器缓存中读取数据并写出（无需再读全局内存）
    int idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = (reg_cache[idx] = row_max) * inv_sum;
        idx ++;
    }
}

// V4: Grid Stride 多行处理 + 所有优化集成
__global__ void online_softmax_v4(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        float* x = input  + row * N;
        float* y = output + row * N;

        float4* x4 = reinterpret_cast<float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        int N4 = N / 4;


        // 第 1 遍：向量化 Online 递推
        float local_m = -INFINITY;
        float local_d = 0.0f;
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            // 对 4 个元素依次更新 (m, d)
            float vals[4] = {data.x, data.y, data.z, data.w};
            for (int k = 0; k < 4; k ++) {
                float m_new = fmax(local_m, vals[k]);
                local_d = local_d * expf(local_m - m_new) + exp(vals[k] - m_new);
                local_m = m_new;
            }
        }
        // 尾部处理
        for (int i = N4 * 4 + tid; i < N; i ++) {
            float xi = x[i];
            float m_new = fmax(local_m, xi);
            local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
            local_m = m_new;
        }
        // 两级 Warp Shuffle 合并（与 V1 相同）
        warpReduceOnline(local_m, local_d);

        // Warp 间通过 Shared Memory 交换
        __shared__ float warp_m[32];
        __shared__ float warp_d[32];
        if (lane == 0) {
            warp_m[wid] = local_m;
            warp_d[wid] = local_d;
        }
        __syncthreads();

        // 第二级：Warp 0 做最终合并
        int num_warps = blockDim.x / 32;
        if (wid == 0) {
            local_m = lane < num_warps ? warp_m[lane] : -INFINITY;
            local_d = lane < num_warps ? warp_d[lane] : 0.0f;
            warpReduceOnline(local_m, local_d);
        }

        // 广播最终结果
        __shared__ float final_m, final_d;
        if (tid == 0) {
            final_m = local_m;
            final_d = local_d;
        }
        __syncthreads();

        float row_max = final_m;
        float inv_sum = 1.0 / final_d;

        // 第 2 遍：向量化归一化
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            float4 result;
            result.x = (data.x - row_max) * inv_sum;
            result.y = (data.y - row_max) * inv_sum;
            result.z = (data.z - row_max) * inv_sum;
            result.w = (data.w - row_max) * inv_sum;
            y4[i] = result;
        }

        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            y[i] = exp(x[i] - row_max) * inv_sum;
        }
        __syncthreads();  // 确保 Shared Memory 在下一行之前被重置；防止广播的问题
    }
}

int main() {

    return 0;
}