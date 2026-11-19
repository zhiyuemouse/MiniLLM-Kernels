#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

void save_f32(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    file.close();
    if (!file) {
        throw std::runtime_error("无法写入 " + path);
    }
}

__global__ void row_softmax_kernel(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    __shared__ float smem[256];
    // 第一遍扫描找最大值
    float max_val = -INFINITY;
    for (int i =  tid; i < N; i += blockDim.x) {
        max_val = fmax(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] = fmax(smem[tid], smem[tid + step]);
        }
        __syncthreads();
    }
    __shared__ float row_max;
    if (tid == 0) row_max = smem[0];
    __syncthreads();

    // 第二遍扫描求和
    float sum = 0.0f;
    for (int i =  tid; i < N; i += blockDim.x) {
        sum += expf(x[i] - row_max);
    }
    smem[tid] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    __shared__ float final_sum;
    if (tid == 0) final_sum = smem[0];
    __syncthreads();

    // 第三遍扫描求输出
    float inv_sum = 1.0 / final_sum;
    for (int i =  tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) * inv_sum;
    }
}

__global__ void row_softmax_cached_exp_kernel(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    __shared__ float smem[256];
    // 第一遍扫描找最大值
    float max_val = -INFINITY;
    for (int i =  tid; i < N; i += blockDim.x) {
        max_val = fmax(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] = fmax(smem[tid], smem[tid + step]);
        }
        __syncthreads();
    }
    __shared__ float row_max;
    if (tid == 0) row_max = smem[0];
    __syncthreads();
    // 第二遍扫描求和
    float reg_cache[8];
    float sum = 0.0f;
    int count = 0;
    for (int i =  tid; i < N; i += blockDim.x) {
        float data = expf(x[i] - row_max);
        reg_cache[count] = data;
        count ++;
        sum += data;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    __shared__ float final_sum;
    if (tid == 0) final_sum = smem[0];
    __syncthreads();

    // 求输出(直接从局部缓存读取)
    float inv_sum = 1.0 / final_sum;
    int idx = 0;
    for (int i =  tid; i < N; i += blockDim.x) {
        y[i] = reg_cache[idx] * inv_sum;
        idx ++;
    }
}

__global__ void row_softmax_cached_exp_unrolled_kernel(float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    __shared__ float smem[256];
    // 第一遍扫描找最大值
    float max_val = -INFINITY;
    for (int i =  tid; i < N; i += blockDim.x) {
        max_val = fmax(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] = fmax(smem[tid], smem[tid + step]);
        }
        __syncthreads();
    }
    __shared__ float row_max;
    if (tid == 0) row_max = smem[0];
    __syncthreads();
    // 第二遍扫描求和
    float reg_cache[8];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float data = expf(x[tid + k * 256] - row_max);
        reg_cache[k] = data;
        sum += data;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    __shared__ float final_sum;
    if (tid == 0) final_sum = smem[0];
    __syncthreads();

    // 求输出(直接从局部缓存读取)
    float inv_sum = 1.0f / final_sum;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        y[tid + k * 256] = reg_cache[k] * inv_sum;
    }
}

void test_row_softmax() {
    struct Shape {
        int rows;
        int cols;
    };
    const Shape shapes[] = {{1, 1}, {3, 17}, {4, 256}, {5, 257}, {32, 2048}};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    constexpr int threads = 256;
    for (const auto& shape : shapes) {
        const auto [rows, cols] = shape;
        const std::size_t count = static_cast<std::size_t>(rows) * cols;
        const std::size_t bytes = count * sizeof(float);
        std::vector<float> random_input(count);
        for (float& value : random_input) {
            value = dist(rng);
        }
        std::vector<float> input(count);
        std::vector<float> output(count);
        float* d_input = nullptr;
        float* d_output = nullptr;
        check_cuda(cudaMalloc(&d_input, bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(&d_output, bytes), "cudaMalloc output");
        for (const std::string mode : {"ones", "random", "negative", "shifted"}) {
            for (std::size_t i = 0; i < count; ++i) { // 根据不同模式初始化
                const float value = random_input[i];
                if (mode == "ones") {
                    input[i] = 1.0f;
                } else if (mode == "negative") {
                    input[i] = -std::abs(value) - 1.0f;
                } else if (mode == "shifted") {
                    input[i] = value + 1000.0f;
                } else {
                    input[i] = value;
                }
            }
            check_cuda(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
            row_softmax_kernel<<<rows, threads>>>(d_input, d_output, rows, cols);
            check_cuda(cudaGetLastError(), "launch row_softmax");
            check_cuda(cudaDeviceSynchronize(), "execute row_softmax");
            check_cuda(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "copy output");
            const std::string name = mode + "_" + std::to_string(rows) + "x" + std::to_string(cols);
            save_f32("input_" + name + ".bin", input);
            save_f32("output_" + name + ".bin", output);
            std::cout << "saved " << name << '\n';
        }
        check_cuda(cudaFree(d_output), "cudaFree output");
        check_cuda(cudaFree(d_input), "cudaFree input");
    }
}

template <typename Launch>
float time_kernel_batch(Launch launch, cudaStream_t stream, int repeats) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    for (int i = 0; i < 10; i ++) { // 往同一 stream 排入 10 次原版 launch
        launch(stream);
        check_cuda(cudaGetLastError(), "check error");
    }

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

template <typename Launch1, typename Launch2>
std::vector<float> time_kernel_pair(Launch1 launch1, Launch2 launch2, cudaStream_t stream, int repeats) {
    cudaEvent_t start1{};
    cudaEvent_t stop1{};
    cudaEvent_t start2{};
    cudaEvent_t stop2{};
    check_cuda(cudaEventCreate(&start1), "create start event");
    check_cuda(cudaEventCreate(&stop1), "create stop event");
    check_cuda(cudaEventCreate(&start2), "create start event");
    check_cuda(cudaEventCreate(&stop2), "create stop event");
    for (int i = 0; i < 10; i ++) { // 往同一 stream 排入 10 次原版 launch
        launch1(stream);
        check_cuda(cudaGetLastError(), "check error");
    }
    check_cuda(cudaEventRecord(start1, stream), "record start event");
    for (int i = 0; i < repeats; i ++) {
        launch1(stream);
    }
    check_cuda(cudaEventRecord(stop1, stream), "record stop event");
    
    for (int i = 0; i < 10; i ++) { // 往同一 stream 排入 10 次原版 launch
        launch2(stream);
        check_cuda(cudaGetLastError(), "check error");
    }
    check_cuda(cudaEventRecord(start2, stream), "record start event");
    for (int i = 0; i < repeats; i ++) {
        launch2(stream);
    }
    check_cuda(cudaEventRecord(stop2, stream), "record stop event");
    
    check_cuda(cudaEventSynchronize(stop2), "wait");
    float elapsed_ms1;
    float elapsed_ms2;
    check_cuda(cudaEventElapsedTime(&elapsed_ms1, start1, stop1), "calculate elapsed_ms");
    check_cuda(cudaEventElapsedTime(&elapsed_ms2, start2, stop2), "calculate elapsed_ms");

    check_cuda(cudaEventDestroy(start1), "destroy start event");
    check_cuda(cudaEventDestroy(stop1), "destroy stop event");
    check_cuda(cudaEventDestroy(start2), "destroy start event");
    check_cuda(cudaEventDestroy(stop2), "destroy stop event");
    return {elapsed_ms1 / repeats, elapsed_ms2 / repeats}; // 返回ms
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

void benchmark_row_softmax() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const int rows = 4096;
    const int cols = 2048;
    constexpr int threads = 256;
    std::vector<float> input(rows * cols);
    std::vector<float> output(rows * cols);
    const size_t bytes = rows * cols * sizeof(float);
    for (float& value : input) {
        value = dist(rng);
    }
    float* d_input = nullptr;
    float* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, bytes), "malloc for d_input");
    check_cuda(cudaMalloc(&d_output, bytes), "malloc for d_output");
    check_cuda(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy data to cuda");

    auto launch_original = [&](cudaStream_t stream) {
        row_softmax_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
    };
    auto launch_cachedexp = [&](cudaStream_t stream) {
        // row_softmax_cached_exp_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
        row_softmax_cached_exp_unrolled_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
        // row_softmax_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
    };
    cudaStream_t stream = 0;
    // 预热20次
    for (int i = 0; i < 20; i ++) {
        time_kernel_batch(launch_original, stream, 1);
        check_cuda(cudaGetLastError(), "check error");
        time_kernel_batch(launch_cachedexp, stream, 1);
        check_cuda(cudaGetLastError(), "check error");
    }
    check_cuda(cudaStreamSynchronize(stream), "finish softmax warmup");
    // 正式采样
    float elapsed_ms_original;
    float elapsed_ms_cachedexp;
    std::vector<float> original_elapsed_list;
    std::vector<float> cachedexp_elapsed_list;
    std::vector<float> divList_originalFist;
    std::vector<float> divList_cachedexpFist;
    constexpr int pairs = 20;
    std::vector<bool> shuffle_first(pairs, false);
    std::fill_n(shuffle_first.begin(), pairs / 2, true);

    std::mt19937 order_rng(42);
    std::shuffle(shuffle_first.begin(), shuffle_first.end(), order_rng);
    for (int i = 0; i < 20; i ++) { // 采样多少轮
        std::string test_mode;
        if (shuffle_first[i]) {
            test_mode = "original first, cachedexp second";
            // elapsed_ms_original = time_kernel_batch(launch_original, stream, 100);
            // check_cuda(cudaGetLastError(), "check error");
            // elapsed_ms_cachedexp = time_kernel_batch(launch_cachedexp, stream, 100);
            // check_cuda(cudaGetLastError(), "check error");
            // divList_originalFist.push_back(elapsed_ms_original - elapsed_ms_cachedexp);
            std::vector<float> ans = time_kernel_pair(launch_original, launch_original, stream, 100);
            elapsed_ms_original = ans[1];
            ans = time_kernel_pair(launch_original, launch_cachedexp, stream, 100);
            elapsed_ms_cachedexp = ans[1];
            divList_originalFist.push_back(elapsed_ms_original - elapsed_ms_cachedexp);
        } else {
            test_mode = "cachedexp first, original second";
            // elapsed_ms_cachedexp = time_kernel_batch(launch_cachedexp, stream, 100);
            // check_cuda(cudaGetLastError(), "check error");
            // elapsed_ms_original = time_kernel_batch(launch_original, stream, 100);
            // check_cuda(cudaGetLastError(), "check error");
            // divList_cachedexpFist.push_back(elapsed_ms_original - elapsed_ms_cachedexp);
            std::vector<float> ans = time_kernel_pair(launch_original, launch_cachedexp, stream, 100);
            elapsed_ms_cachedexp = ans[1];
            ans = time_kernel_pair(launch_original, launch_original, stream, 100);
            elapsed_ms_original = ans[1];
            divList_cachedexpFist.push_back(elapsed_ms_original - elapsed_ms_cachedexp);
        }
        std::cout << "批次:" << i + 1 << " | " << "测试顺序:" << test_mode << " | " << "elapsed_ms_original(ms):" << elapsed_ms_original << " | " << "elapsed_ms_cachedexp(ms):" << elapsed_ms_cachedexp << std::endl;
        original_elapsed_list.push_back(elapsed_ms_original);
        cachedexp_elapsed_list.push_back(elapsed_ms_cachedexp);
    }
    std::sort(original_elapsed_list.begin(), original_elapsed_list.end()); // shuffle 耗时
    std::sort(cachedexp_elapsed_list.begin(), cachedexp_elapsed_list.end()); // contiguous 耗时
    std::sort(divList_originalFist.begin(), divList_originalFist.end()); // shuffle fist的差值
    std::sort(divList_cachedexpFist.begin(), divList_cachedexpFist.end()); // contiguous fist的差值
    float elapsed_original_med = getMedium(original_elapsed_list); // shuffle 耗时中位数
    float elapsed_cachedexp_med = getMedium(cachedexp_elapsed_list); // // contiguous 耗时中位数
    float originalFist_div_med = getMedium(divList_originalFist); // shuffle fist的差值中位数
    float cachedexpFist_div_med = getMedium(divList_cachedexpFist); // // contiguous fist的差值中位数
    std::cout << "originalFist 的差值的中位数:" << originalFist_div_med << std::endl;
    std::cout << "cachedexpFist first 的差值的中位数:" << cachedexpFist_div_med << std::endl;
    std::cout << "elapsed_original中位数(ms):" << elapsed_original_med << " | " << "elapsed_cachedexp(ms):" << elapsed_cachedexp_med << std::endl;
    std::cout << "original是cachedexp的:" << elapsed_original_med / elapsed_cachedexp_med << "倍" << std::endl;

    // 保存
    row_softmax_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_f32("output_original.bin", output);

    // row_softmax_cached_exp_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
    row_softmax_cached_exp_unrolled_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
    // row_softmax_kernel<<<rows, threads, 0, stream>>>(d_input, d_output, rows, cols);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_f32("output_cachedexp.bin", output);
    
    save_f32("input.bin", input);
    check_cuda(cudaFree(d_input), "free d_input");
    check_cuda(cudaFree(d_output), "free d_output");

}

int main() {
    bool benchmark = true;
    if (benchmark) benchmark_row_softmax();
    else test_row_softmax();

    return 0;
}