cmake_minimum_required(VERSION 3.16)
project(problem764 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(problem764 src/main.cu src/cpu/problem764_cpu.cu src/cpu/problem764_cpu.cu src/result.cu src/utils.cu src/cuda/problem764_cuda.cu)

set_target_properties(problem764 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
