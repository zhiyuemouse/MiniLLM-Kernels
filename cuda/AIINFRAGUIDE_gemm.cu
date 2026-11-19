#include <stdio.h>

// 朴素实现：建立 Baseline
__global__ void sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k ++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Thread Block 级 Tiling 优化
template <int BM, int BN, int BK, int BLOCK_SIZE> // 这里假设输入为 (128, 8), (8, 128), BLOCK_SIZE=256
__global__ void sgemm_block_tiling(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK; // =8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // =32
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

    // 计算 tileC 时的线程排布（16×16）
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X; // =16
    int c_thread_x = tid % C_BLOCK_X;
    int c_thread_y = tid / C_BLOCK_X;

    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y; // =8
    constexpr int Tn = BN / C_BLOCK_X; // =8
    float Ct[Tm][Tn] = {0.0f};

    // K-Loop
    for (int k = 0; k < K; k += BK) {
        // 协作加载 tileA（使用跨步循环覆盖 BM 行）
        #pragma unroll // 展开循环
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int r = r0 + i, c = k + a_thread_x;
            As[i][a_thread_x] = (r < M && c < N) ? A[r * K + c] : 0.0f;
        }

        // 协作加载 tileB（使用跨步循环覆盖 BN 列）
        #pragma unroll // 展开循环
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = k + b_thread_y, c = c0 + j;
            Bs[b_thread_y][j] = (r < M && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

        // 外积方式计算 As × Bs
        #pragma unroll
        for (int p = 0; p < Bk; p ++) {
            for (int i = 0; i < Tm; i ++) {
                int row = c_thread_y + C_BLOCK_Y * i;
                for (int j = 0; j < Tn; j ++) {
                    int col = c_thread_x + C_BLOCK_X * j;
                    Ct[i][j] += As[row][p] * Bs[p][col];
                }
            }
        }
        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < Tm; i ++) {
        int r = r0 + c_thread_y + C_BLOCK_Y * i;
        for (int j = 0; j < Tn; j ++) {
            int c = c0 + c_thread_x + C_BLOCK_X * j;
            if (r < M && c < N) C[r][c] = Ct[i][j];
        }
    }
}

// Thread 级 Tiling 优化：寄存器级数据复用
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个线程在 C 中负责 TM×TN 的子块
    const int thread_num = BM / TM * BN / TN; // =256
    int tid = threadIdx.x;
    int thread_row = (tid / (BN / TN)) * TM;
    int thread_col = (tid % (BN / TN)) * TN;

    // 寄存器存储
    float a_frag[TM];
    float b_frag[TN];
    flaot c_frag[TM][TN] = {0.0f};

    int by = blockIdx.y, bx = blockIdx.x;

    for (int bk = 0; bk < BK; bk += BK) {
        // 协作加载 A、B 到 Shared Memory（省略边界检查）
        load_tile_A(A, As, by, bk, tid, M, K);
        load_tile_B(B, Bs, bx, bk, tid, K, N);
        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k ++) {
            // 从 Shared Memory 加载到寄存器
            for (int i = 0; i < TM; i ++)
                a_frag[i] = As[thread_row + i][k];
            for (int j = 0; j < TN; j ++)
                b_frag[j] = Bs[k][thread_col + j];
            // 寄存器上做外积
            for (int i = 0; i < TM; i ++)
                for (int j = 0; j < TN; j ++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }
    // 写回 Global Memory
    for (int i = 0; i < TM; i ++)
        for (int j = 0; j < TN; j ++)
            C[(by * BM + thread_row + i) * N + (bx * BN + thread_col + j)] = c_frag[i][j];
}

int main() {

    return 0;
}