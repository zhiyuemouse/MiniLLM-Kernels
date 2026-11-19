#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <type_traits>
#include <random>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>

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

template <typename T>
__global__ void row_sum_kernel(
    const T* input,
    float* output,
    int rows,
    int cols
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    float val = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        val += static_cast<float>(input[row * cols + col]);
    }
    smem[tid] = val;
    __syncthreads();
    for (unsigned int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (step * 2) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
};

template <typename T>
__global__ void row_sum_contiguous_kernel(
    const T* input,
    float* output,
    int rows,
    int cols
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    float val = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        val += static_cast<float>(input[row * cols + col]);
    }
    smem[tid] = val;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
};

template <typename T>
__global__ void row_sum_warp_shuffle_kernel(
    const T* input,
    float* output,
    int rows,
    int cols
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int wid = tid / 32;
    int lane_id = tid % 32;
    float val = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        val += static_cast<float>(input[row * cols + col]);
    }
    // Warp 内归约
    for (unsigned int step = 16; step > 0; step >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, step);
    }
    if (lane_id == 0) {
        smem[wid] = val;
    }
    __syncthreads();
    // Warp 内归约
    if (wid == 0) {
        val = lane_id < blockDim.x / 32? smem[lane_id]: 0.0f;
        for (unsigned int step = 16; step > 0; step >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, step);
        }
    }
    
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
};

auto check_cuda = [](cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
};

template <typename T>
void test_row_sum(T matric_val, bool useRandom, std::string useMode) {
    std::mt19937 rng(42); // 固定随机种子
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // 随机数创建
    std::vector<int> allRows = {1, 3, 4, 5, 32};
    std::vector<int> allCols = {1, 17, 256, 257, 2048};
    for (int test_id = 0; test_id < allRows.size(); test_id ++) {
        std::cout << "====================================" << "test:" << test_id + 1 << "====================================" << std::endl;
        int rows = allRows[test_id];
        int cols = allCols[test_id];
        std::vector<T> input(rows * cols);
        std::vector<float> output(rows);
        // 初始化输入矩阵
        for (int i = 0; i < rows; i ++) {
            // std::cout << "Line:" << i + 1 << std::endl;
            for (int j = 0; j < cols; j ++) {
                if constexpr(std::is_same_v<T, __half>) {
                    if (useRandom) input[i * cols + j] = __float2half(dist(rng));
                    else input[i * cols + j] = __float2half(matric_val);
                } else {
                    if (useRandom) input[i * cols + j] = dist(rng);
                    else input[i * cols + j] = matric_val;
                }
                // std::cout << i * cols + j << " ";
            }
            // std::cout << std::endl;
        }
        T* d_input = nullptr;
        float* d_output = nullptr;
        const size_t input_bytes = input.size() * sizeof(T);
        const size_t output_bytes = output.size() * sizeof(float);
        check_cuda(cudaMalloc(&d_input, input_bytes), "allocate input");
        check_cuda(cudaMalloc(&d_output, output_bytes), "allocate output");
        check_cuda(
            cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice),
            "copy input to GPU"
        );


        constexpr int threads = 256;
        if (useMode == "contiguous") row_sum_contiguous_kernel<T><<<rows, threads, threads * sizeof(float)>>>(d_input, d_output, rows, cols);
        else if(useMode == "shuffle") row_sum_warp_shuffle_kernel<T><<<rows, threads, (threads / 32) * sizeof(float)>>>(d_input, d_output, rows, cols); // 按 warp 数计算申请量
        else row_sum_kernel<T><<<rows, threads, threads * sizeof(float)>>>(d_input, d_output, rows, cols);
        check_cuda(cudaGetLastError(), "launch row_sum");
        check_cuda(cudaDeviceSynchronize(), "execute row_sum");
        
        check_cuda(
            cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost),
            "copy output to CPU"
        );
        check_cuda(cudaFree(d_input), "free input");
        check_cuda(cudaFree(d_output), "free output");
        for (int i = 0; i < rows; i ++) {
            std::cout << "sum:" << output[i] << std::endl;
        }
        std::string input_path;
        std::string output_path;
        if (useRandom) {
            input_path = "input_random";
            output_path = "output_random";
        } else {
            input_path = "input_val";
            output_path = "output_val";
        }
        if constexpr(std::is_same_v<T, __half>) {
            input_path += "_half";
            output_path += "_half";
        } else {
            input_path += "_float";
            output_path += "_float";
        }
        const std::string shape_suffix =
            "_" + std::to_string(rows) + "x" + std::to_string(cols);
        save_float_bin(input_path + shape_suffix + ".bin", input);
        save_float_bin(output_path + shape_suffix + ".bin", output);

        // 原文件名继续保存最后一组尺寸，兼容已有读取方式。
        if (test_id + 1 == allRows.size()) {
            save_float_bin(input_path + ".bin", input);
            save_float_bin(output_path + ".bin", output);
        }
    }  
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

template <typename T>
void benchmark_row_sum() {
    std::mt19937 rng(42); // 固定随机种子
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // 随机数创建

    int rows = 4096;
    int cols = 2048;
    std::vector<T> input(rows * cols);
    std::vector<float> output(rows);
    // 初始化输入矩阵
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            if constexpr(std::is_same_v<T, __half>) {
                input[i * cols + j] = __float2half(dist(rng));
            } else {
                input[i * cols + j] = dist(rng);
            }
        }
    }
    T* d_input = nullptr;
    float* d_output = nullptr;
    const size_t input_bytes = input.size() * sizeof(T);
    const size_t output_bytes = output.size() * sizeof(float);
    check_cuda(cudaMalloc(&d_input, input_bytes), "allocate input");
    check_cuda(cudaMalloc(&d_output, output_bytes), "allocate output");
    check_cuda(
        cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice),
        "copy input to GPU"
    );


    constexpr int threads = 256;
    cudaStream_t stream = 0;
    auto launch_shuffle = [&](cudaStream_t stream) {
        row_sum_warp_shuffle_kernel<T><<<rows, threads, (threads / 32) * sizeof(float), stream>>>(d_input, d_output, rows, cols);
    };
    auto launch_contiguous = [&](cudaStream_t stream) {
        row_sum_contiguous_kernel<T><<<rows, threads, threads * sizeof(float), stream>>>(d_input, d_output, rows, cols);
    };
    float elapsed_ms_shuffle;
    float elapsed_ms_contiguous;
    // 预热 20 次
    for (int i = 0; i < 20; i ++) {
        elapsed_ms_shuffle = time_kernel_batch(launch_shuffle, 0, 100);
        check_cuda(cudaGetLastError(), "launch row_sum");
        check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    }
    for (int i = 0; i < 20; i ++) {
        elapsed_ms_contiguous = time_kernel_batch(launch_contiguous, 0, 100);
        check_cuda(cudaGetLastError(), "launch row_sum");
        check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    }
    // 正式采样
    std::vector<float> shuffle_elapsed_list;
    std::vector<float> contiguous_elapsed_list;
    std::vector<float> shuffle_contiguous_div_list_shuffleFist;
    std::vector<float> shuffle_contiguous_div_list_contiguousFist;
    constexpr int pairs = 20;
    std::vector<bool> shuffle_first(pairs, false);
    std::fill_n(shuffle_first.begin(), pairs / 2, true);

    std::mt19937 order_rng(42);
    std::shuffle(shuffle_first.begin(), shuffle_first.end(), order_rng);
    for (int i = 0; i < 20; i ++) { // 采样多少轮
        std::string test_mode;
        if (shuffle_first[i]) {
            test_mode = "shuffle first, contiguous second";
            elapsed_ms_shuffle = time_kernel_batch(launch_shuffle, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            elapsed_ms_contiguous = time_kernel_batch(launch_contiguous, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            shuffle_contiguous_div_list_shuffleFist.push_back(elapsed_ms_contiguous - elapsed_ms_shuffle);
        } else {
            test_mode = "contiguous first, shuffle second";
            elapsed_ms_contiguous = time_kernel_batch(launch_contiguous, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            elapsed_ms_shuffle = time_kernel_batch(launch_shuffle, 0, 100);
            check_cuda(cudaGetLastError(), "launch row_sum");
            check_cuda(cudaDeviceSynchronize(), "execute row_sum");
            shuffle_contiguous_div_list_contiguousFist.push_back(elapsed_ms_contiguous - elapsed_ms_shuffle);
        }
        std::cout << "批次:" << i + 1 << " | " << "测试顺序:" << test_mode << " | " << "elapsed_ms_shuffle(ms):" << elapsed_ms_shuffle << " | " << "elapsed_ms_contiguous(ms):" << elapsed_ms_contiguous << std::endl;

        shuffle_elapsed_list.push_back(elapsed_ms_shuffle);
        contiguous_elapsed_list.push_back(elapsed_ms_contiguous);
    }
    std::sort(shuffle_elapsed_list.begin(), shuffle_elapsed_list.end()); // shuffle 耗时
    std::sort(contiguous_elapsed_list.begin(), contiguous_elapsed_list.end()); // contiguous 耗时
    std::sort(shuffle_contiguous_div_list_shuffleFist.begin(), shuffle_contiguous_div_list_shuffleFist.end()); // shuffle fist的差值
    std::sort(shuffle_contiguous_div_list_contiguousFist.begin(), shuffle_contiguous_div_list_contiguousFist.end()); // contiguous fist的差值
    float elapsed_shuffle_med = getMedium(shuffle_elapsed_list); // shuffle 耗时中位数
    float elapsed_contiguous_med = getMedium(contiguous_elapsed_list); // // contiguous 耗时中位数
    float shuffleFist_div_med = getMedium(shuffle_contiguous_div_list_shuffleFist); // shuffle fist的差值中位数
    float contiguousFist_div_med = getMedium(shuffle_contiguous_div_list_contiguousFist); // // contiguous fist的差值中位数
    std::cout << "shuffle first 的差值的中位数:" << shuffleFist_div_med << std::endl;
    std::cout << "contiguous first 的差值的中位数:" << contiguousFist_div_med << std::endl;
    std::cout << "elapsed_ms_shuffle中位数(ms):" << elapsed_shuffle_med << " | " << "elapsed_ms_contiguous中位数(ms):" << elapsed_contiguous_med << std::endl;
    std::cout << "contiguous是shuffle的:" << elapsed_contiguous_med / elapsed_shuffle_med << "倍" << std::endl;
    // 保存original_output.bin
    row_sum_kernel<T><<<rows, threads, threads * sizeof(float), stream>>>(d_input, d_output, rows, cols);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_float_bin("original_output.bin", output);
    // 保存contiguous_output.bin
    row_sum_contiguous_kernel<T><<<rows, threads, threads * sizeof(float), stream>>>(d_input, d_output, rows, cols);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_float_bin("contiguous_output.bin", output);
    // 保存shuffle_output
    row_sum_warp_shuffle_kernel<T><<<rows, threads, (threads / 32) * sizeof(float), stream>>>(d_input, d_output, rows, cols);
    check_cuda(cudaGetLastError(), "launch row_sum");
    check_cuda(cudaDeviceSynchronize(), "execute row_sum");
    check_cuda(
        cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost),
        "copy output to CPU"
    );
    save_float_bin("shuffle_output.bin", output);

    save_float_bin("benchmark_input.bin", input);
    check_cuda(cudaFree(d_input), "free input");
    check_cuda(cudaFree(d_output), "free output");
}

int main() {
    bool benchmark = true;
    if (benchmark) {
        benchmark_row_sum<float>();
    } else {
        std::cout << "###########################" <<"float test" << "###########################" << std::endl;
        test_row_sum<float>(1, false, "shuffle"); // 测试全1输入
        test_row_sum<float>(1, true, "shuffle"); // 测试全随机输入
        std::cout << "###########################" <<"__half test" << "###########################" << std::endl;
        test_row_sum<__half>(1, false, "shuffle"); // 测试全1输入
        test_row_sum<__half>(1, true, "shuffle"); // 测试全随机输入
    }
    
    return 0;
}
