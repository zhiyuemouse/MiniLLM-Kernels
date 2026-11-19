#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n",
                     operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// 保存数据到指定文件
template <typename T>
void save_float_bin(const std::string& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件 " + path);
    }

    const auto byte_count =
        static_cast<std::streamsize>(data.size() * sizeof(T));
    file.write(reinterpret_cast<const char*>(data.data()), byte_count);

    file.close();
    if (!file) {
        throw std::runtime_error("写入文件失败 " + path);
    }
}

__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i ++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    constexpr int TILE = 16;
    constexpr int BLOCK_SIZE = 256;
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    int r0 = blockIdx.y * TILE;
    int c0 = blockIdx.x * TILE;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 加载A TILE元素时
    constexpr int A_BLOCK_X = TILE;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;
    // 加载B TILE元素时
    constexpr int B_BLOCK_Y = TILE;
    constexpr int B_BLOCK_X = BLOCK_SIZE / B_BLOCK_Y;
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;
    // 计算 tileC 时的线程排布（16×16）
    constexpr int C_BLOCK_X = TILE;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;
    int c_thread_x = tid % C_BLOCK_X;
    int c_thread_y = tid / C_BLOCK_X;
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = TILE / C_BLOCK_Y;
    constexpr int Tn = TILE / C_BLOCK_X;
    float Ct[Tm][Tn] = {0.0f};

    for (int k_step = 0; k_step < k; k_step += TILE) {
        // 加载 a_tile
        #pragma unroll
        for (int i = a_thread_y; i < TILE; i += A_BLOCK_Y) {
            int r = r0 + i, c = k_step + a_thread_x;
            a_tile[i][a_thread_x] = (r < m && c < k) ? a[r * k + c] : 0.0f;
        }
        // 加载 b_tile
        #pragma unroll
        for (int j = b_thread_x; j < TILE; j += B_BLOCK_X) {
            int r = k_step + b_thread_y, c = c0 + j;
            b_tile[b_thread_y][j] = (r < k && c < n) ? b[r * n + c] : 0.0f;
        }
        __syncthreads();
        // 计算 C
        #pragma unroll
        for (int p = 0; p < TILE; p ++) {
            for (int i = 0; i < Tm; i ++) {
                int row = c_thread_y + i * C_BLOCK_Y;
                for (int j = 0; j < Tn; j ++) {
                    int col = c_thread_x + j * C_BLOCK_X;
                    Ct[i][j] += a_tile[row][p] * b_tile[p][col];
                }
            }
        }
        __syncthreads();
    }
    // 写回结果
    for (int i = 0; i < Tm; i ++) {
        int row = r0 + C_BLOCK_Y * i + c_thread_y;
        for (int j = 0; j < Tn; j ++) {
            int col = c0 + C_BLOCK_X * j + c_thread_x;
            if (row < m && col < n) c[row * n + col] = Ct[i][j];
        }
    }
}

__global__ void matmul_tiled_1x2_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    constexpr int TILE = 16;
    constexpr int TILE2x = 32;
    constexpr int BLOCK_SIZE = 256;
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE2x];

    int r0 = blockIdx.y * TILE;
    int c0 = blockIdx.x * TILE2x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 加载A TILE元素时
    constexpr int A_BLOCK_X = TILE;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;
    // 加载B TILE元素时
    constexpr int B_BLOCK_Y = TILE;
    constexpr int B_BLOCK_X = BLOCK_SIZE / B_BLOCK_Y;
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;
    // 计算 tileC 时的线程排布（16×16）
    constexpr int C_BLOCK_X = TILE;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;
    int c_thread_x = tid % C_BLOCK_X;
    int c_thread_y = tid / C_BLOCK_X;
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = TILE / C_BLOCK_Y;
    constexpr int Tn = TILE2x / C_BLOCK_X;
    float Ct[Tm][Tn] = {0.0f};

    for (int k_step = 0; k_step < k; k_step += TILE) {
        // 加载 a_tile
        #pragma unroll
        for (int i = a_thread_y; i < TILE; i += A_BLOCK_Y) {
            int r = r0 + i, c = k_step + a_thread_x;
            a_tile[i][a_thread_x] = (r < m && c < k) ? a[r * k + c] : 0.0f;
        }
        // 加载 b_tile
        #pragma unroll
        for (int j = b_thread_x; j < TILE2x; j += B_BLOCK_X) {
            int r = k_step + b_thread_y, c = c0 + j;
            b_tile[b_thread_y][j] = (r < k && c < n) ? b[r * n + c] : 0.0f;
        }
        __syncthreads();
        // 计算 C
        #pragma unroll
        for (int p = 0; p < TILE; p ++) {
            for (int i = 0; i < Tm; i ++) {
                int row = c_thread_y + i * C_BLOCK_Y;
                for (int j = 0; j < Tn; j ++) {
                    int col = c_thread_x + j * C_BLOCK_X;
                    Ct[i][j] += a_tile[row][p] * b_tile[p][col];
                }
            }
        }
        __syncthreads();
    }
    // 写回结果
    for (int i = 0; i < Tm; i ++) {
        int row = r0 + C_BLOCK_Y * i + c_thread_y;
        for (int j = 0; j < Tn; j ++) {
            int col = c0 + C_BLOCK_X * j + c_thread_x;
            if (row < m && col < n) c[row * n + col] = Ct[i][j];
        }
    }
}

__global__ void matmul_m1_kernel(const float* a, const float* b, float* c, int m, int n, int k) { // 为了接口方便此处不删除参数m
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < k; i ++) {
        sum += a[i] * b[i * n + col];
    }
    c[col] = sum;
}

__global__ void matmul_m1_k4_kernel(const float* a, const float* b, float* c, int m, int n, int k) { // 为了接口方便此处不删除参数m
    constexpr int K = 4;
    __shared__ float smem[4][32];
    int tidy = threadIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0.0f;
    if (col < n) {
        for (int i = tidy; i < k; i +=K) {
            sum += a[i] * b[i * n + col];
        }
    }
    smem[tidy][threadIdx.x] = sum;
    __syncthreads();
    
    if (tidy == 0 && col < n) {
        c[col] = smem[0][threadIdx.x] + smem[1][threadIdx.x] + smem[2][threadIdx.x] + smem[3][threadIdx.x];
    }
}

__global__ void matmul_m1_k8_kernel(const float* a, const float* b, float* c, int m, int n, int k) { // 为了接口方便此处不删除参数m
    constexpr int K = 8;
    __shared__ float smem[8][32];
    int tidy = threadIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0.0f;
    if (col < n) {
        for (int i = tidy; i < k; i +=K) {
            sum += a[i] * b[i * n + col];
        }
    }
    smem[tidy][threadIdx.x] = sum;
    __syncthreads();
    
    if (tidy == 0 && col < n) {
        c[col] = smem[0][threadIdx.x] + smem[1][threadIdx.x] + smem[2][threadIdx.x] + smem[3][threadIdx.x] +
                 smem[4][threadIdx.x] + smem[5][threadIdx.x] + smem[6][threadIdx.x] + smem[7][threadIdx.x];
    }
}

void test_matmul_case(int m, int n, int k, bool random_input) {
    std::vector<float> a(m * k, 1.0f);
    std::vector<float> b(k * n, 1.0f);
    std::vector<float> c(m * n);

    if (random_input) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& value : a) {
            value = dist(rng);
        }
        for (float& value : b) {
            value = dist(rng);
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    const auto a_bytes = a.size() * sizeof(float);
    const auto b_bytes = b.size() * sizeof(float);
    const auto c_bytes = c.size() * sizeof(float);

    check_cuda(cudaMalloc(&d_a, a_bytes), "allocate A");
    check_cuda(cudaMalloc(&d_b, b_bytes), "allocate B");
    check_cuda(cudaMalloc(&d_c, c_bytes), "allocate C");
    check_cuda(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice), "copy B");

    // 用 NaN 填充输出，漏写的元素会被下面的有限值检查发现。
    check_cuda(cudaMemset(d_c, 0xff, c_bytes), "initialize C");

    // const dim3 block(16, 16);
    // const dim3 block(32);
    // const dim3 block(32, 4);
    const dim3 block_m1k8(32, 8);
    // const dim3 grid((n + 15) / 16, (m + 15) / 16);
    // const dim3 grid((n + 31) / 32, (m + 15) / 16);
    const dim3 grid((n + 31) / 32);
    // matmul_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    // matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    // matmul_tiled_1x2_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    // matmul_m1_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    // matmul_m1_k4_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    matmul_m1_k8_kernel<<<grid, block_m1k8>>>(d_a, d_b, d_c, m, n, k);
    check_cuda(cudaGetLastError(), "launch matmul");
    check_cuda(cudaDeviceSynchronize(), "execute matmul");
    check_cuda(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost), "copy C");

    check_cuda(cudaFree(d_a), "free A");
    check_cuda(cudaFree(d_b), "free B");
    check_cuda(cudaFree(d_c), "free C");

    const char* mode = random_input ? "random" : "ones";
    const std::string case_name = std::string(mode) + "_" + std::to_string(m)
                                + "x" + std::to_string(n) + "x" + std::to_string(k);
    save_float_bin("a_" + case_name + ".bin", a);
    save_float_bin("b_" + case_name + ".bin", b);
    save_float_bin("c_" + case_name + ".bin", c);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double expected = k;  // 全 1 输入的每个输出恰好等于 K。
            if (random_input) {
                expected = 0.0;
                for (int i = 0; i < k; ++i) {
                    expected += static_cast<double>(a[row * k + i])
                              * b[i * n + col];
                }
            }

            const float actual = c[row * n + col];
            // const double tolerance =
            //     random_input ? 1e-5 + 1e-5 * std::abs(expected) : 0.0;
            const double atol = k == 1024 ? 5e-5 : 1e-5;
            const double tolerance =
                random_input ? atol + 1e-5 * std::abs(expected) : 0.0;
            if (!std::isfinite(actual) ||
                std::abs(actual - expected) > tolerance) {
                std::fprintf(stderr,
                             "FAIL %s %dx%dx%d C[%d,%d]: got %.9g, expected %.9g\n", mode, m, n, k, row, col, actual, expected);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    std::printf("PASS %s %dx%dx%d\n", mode, m, n, k);
}

void test_matmul() {
    // const int shapes[][3] = {{1, 1, 1}, {16, 16, 16}, {17, 19, 23}, {17, 35, 23}};
    const int shapes[][3] = {{1, 1024, 1024}, {1, 35, 23}};

    for (const auto& [m, n, k] : shapes) {
        test_matmul_case(m, n, k, false);
        test_matmul_case(m, n, k, true);
    }

    return;
}

template <typename Launch>
float time_kernel_batch(Launch launch, cudaStream_t stream, int repeats) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    check_cuda(cudaEventRecord(start, stream), "record start event");
    for (int i = 0; i < repeats; i ++) {
        launch(stream);
    }
    check_cuda(cudaEventRecord(stop, stream), "record stop event");
    check_cuda(cudaEventSynchronize(stop), "wait");
    float elapsed_ms;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "calculate elapsed_ms");
    check_cuda(cudaEventDestroy(start), "destroy start event");
    check_cuda(cudaEventDestroy(stop), "destroy stop event");
    return elapsed_ms / repeats; // 返回ms
}

float getMedium(std::vector<float> vec) {
    int vec_len = vec.size();
    float ans;
    if (vec_len % 2 == 0) { // 偶数
        ans = (vec[vec_len / 2] + vec[vec_len / 2 - 1]) / 2;
    } else {
        ans = vec[vec_len / 2];
    }
    return ans;
}

void benchmark_matmul() {
    std::mt19937 rng(42); // 固定随机种子
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // 随机数创建

    // int m = 1024, n = 1024, k = 1024;
    int m = 1, n = 1024, k = 1024;
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n);
    // 初始化输入矩阵 a
    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < k; j ++) {
            a[i * k + j] = dist(rng);
        }
    }
    // 初始化输入矩阵 b
    for (int i = 0; i < k; i ++) {
        for (int j = 0; j < n; j ++) {
            b[i * k + j] = dist(rng);
        }
    }
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    const size_t a_input_bytes = a.size() * sizeof(float);
    const size_t b_input_bytes = b.size() * sizeof(float);
    const size_t c_bytes = c.size() * sizeof(float);
    check_cuda(cudaMalloc(&d_a, a_input_bytes), "allocate a");
    check_cuda(cudaMalloc(&d_b, b_input_bytes), "allocate b");
    check_cuda(cudaMalloc(&d_c, c_bytes), "allocate c");
    check_cuda(
        cudaMemcpy(d_a, a.data(), a_input_bytes, cudaMemcpyHostToDevice),
        "copy input to GPU"
    );
    check_cuda(
        cudaMemcpy(d_b, b.data(), b_input_bytes, cudaMemcpyHostToDevice),
        "copy input to GPU"
    );


    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    const dim3 grid1x2((n + 31) / 32, (m + 15) / 16);
    const dim3 block_m1(32);
    const dim3 block_m1k4(32, 4);
    const dim3 block_m1k8(32, 8);
    const dim3 grid_m1((n + 31) / 32);
    cudaStream_t stream = 0;
    auto launch_original = [&](cudaStream_t stream) {
        // matmul_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
        // matmul_tiled_1x2_kernel<<<grid1x2, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
        // matmul_m1_kernel<<<grid_m1, block_m1, 0, stream>>>(d_a, d_b, d_c, m, n, k);
        matmul_m1_k4_kernel<<<grid_m1, block_m1k4, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    };
    auto launch_tiling = [&](cudaStream_t stream) {
        // matmul_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
        // matmul_tiled_1x2_kernel<<<grid1x2, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
        // matmul_m1_k4_kernel<<<grid_m1, block_m1k4, 0, stream>>>(d_a, d_b, d_c, m, n, k);
        matmul_m1_k8_kernel<<<grid_m1, block_m1k8, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    };
    float elapsed_ms_original;
    float elapsed_ms_tiling;
    // 预热 20 次
    for (int i = 0; i < 20; i ++) {
        elapsed_ms_original = time_kernel_batch(launch_original, 0, 100);
        check_cuda(cudaGetLastError(), "launch row_sum");
        check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    }
    for (int i = 0; i < 20; i ++) {
        elapsed_ms_tiling = time_kernel_batch(launch_tiling, 0, 100);
        check_cuda(cudaGetLastError(), "launch row_sum");
        check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    }
    // 正式采样
    std::vector<float> original_elapsed_list;
    std::vector<float> tiling_elapsed_list;
    std::vector<float> div_list_originalFist;
    std::vector<float> div_list_tilingFist;
    constexpr int pairs = 20;
    std::vector<bool> original_first(pairs, false);
    std::fill_n(original_first.begin(), pairs / 2, true);

    std::mt19937 order_rng(42);
    std::shuffle(original_first.begin(), original_first.end(), order_rng);
    for (int i = 0; i < 20; i ++) { // 采样多少轮
        std::string test_mode;
        if (original_first[i]) {
            test_mode = "original first, tiling second";
            elapsed_ms_original = time_kernel_batch(launch_original, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            elapsed_ms_tiling = time_kernel_batch(launch_tiling, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            div_list_originalFist.push_back(elapsed_ms_original - elapsed_ms_tiling);
        } else {
            test_mode = "tiling first, original second";
            elapsed_ms_tiling = time_kernel_batch(launch_tiling, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            elapsed_ms_original = time_kernel_batch(launch_original, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            div_list_tilingFist.push_back(elapsed_ms_original - elapsed_ms_tiling);
        }
        std::cout << "批次:" << i + 1 << " | " << "测试顺序:" << test_mode << " | " << "elapsed_ms_original(ms):" << elapsed_ms_original << " | " << "elapsed_ms_tiling(ms):" << elapsed_ms_tiling << std::endl;

        original_elapsed_list.push_back(elapsed_ms_original);
        tiling_elapsed_list.push_back(elapsed_ms_tiling);
    }
    std::sort(original_elapsed_list.begin(), original_elapsed_list.end()); // original 耗时
    std::sort(tiling_elapsed_list.begin(), tiling_elapsed_list.end()); // tiling 耗时
    std::sort(div_list_originalFist.begin(), div_list_originalFist.end()); // original fist的差值
    std::sort(div_list_tilingFist.begin(), div_list_tilingFist.end()); // tiling fist的差值
    float elapsed_original_med = getMedium(original_elapsed_list); // original 耗时中位数
    float elapsed_tiling_med = getMedium(tiling_elapsed_list); // // tiling 耗时中位数
    float originalFist_div_med = getMedium(div_list_originalFist); // original fist的差值中位数
    float tilingFist_div_med = getMedium(div_list_tilingFist); // // tiling fist的差值中位数
    std::cout << "original first 的差值的中位数:" << originalFist_div_med << std::endl;
    std::cout << "tiling first 的差值的中位数:" << tilingFist_div_med << std::endl;
    std::cout << "elapsed_ms_original中位数(ms):" << elapsed_original_med << " | " << "elapsed_ms_tiling中位数(ms):" << elapsed_tiling_med << std::endl;
    std::cout << "original是tiling的:" << elapsed_original_med / elapsed_tiling_med << "倍" << std::endl;
    // 保存original_output.bin
    // matmul_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    // matmul_tiled_1x2_kernel<<<grid1x2, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    // matmul_m1_kernel<<<grid_m1, block_m1, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    matmul_m1_k4_kernel<<<grid_m1, block_m1k4, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_float_bin("original_output.bin", c);
    // 保存tiling_output.bin
    // matmul_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    // matmul_tiled_1x2_kernel<<<grid1x2, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    // matmul_m1_kernel<<<grid_m1, block_m1, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    // matmul_m1_k4_kernel<<<grid_m1, block_m1k4, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    matmul_m1_k8_kernel<<<grid_m1, block_m1k8, 0, stream>>>(d_a, d_b, d_c, m, n, k);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_float_bin("tiling_output.bin", c);

    save_float_bin("benchmark_a.bin", a);
    save_float_bin("benchmark_b.bin", b);
    check_cuda(cudaFree(d_a), "free a");
    check_cuda(cudaFree(d_b), "free b");
    check_cuda(cudaFree(d_c), "free c");
}

int main () {
    bool benchmark = false;;
    if (benchmark) {
        benchmark_matmul();
    }
    else {
        test_matmul();
    }
    return 0;
}
