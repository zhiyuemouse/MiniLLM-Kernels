cmake_minimum_required(VERSION 3.18)
project(MiniLLMKernels LANGUAGES CXX CUDA)

# add_executable(row_sum cuda/reduce/row_sum.cu)
# add_executable(row_softmax cuda/softmax/row_softmax.cu)
add_executable(matmul_naive cuda/gemm/matmul_naive.cu)

set_target_properties(matmul_naive PROPERTIES 
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)

# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
# cmake --build build --parallel
# ../../../build/row_sum