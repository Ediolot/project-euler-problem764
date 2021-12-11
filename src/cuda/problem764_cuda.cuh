//
// Created by jsier on 11/12/2021.
//

#ifndef PROBLEM764_PROBLEM764_CUDA_CUH
#define PROBLEM764_PROBLEM764_CUDA_CUH

#include <cstdint>

constexpr uint64_t calculate_number_blocks(uint64_t m_max);
void cuda_v0(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n);
void cuda_v1(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n);
void cuda_v2(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n, const uint64_t m_max);
void cuda_v3(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n, const uint64_t m_max);
void calculate764_cuda(const uint64_t big_n);

#endif //PROBLEM764_PROBLEM764_CUDA_CUH
