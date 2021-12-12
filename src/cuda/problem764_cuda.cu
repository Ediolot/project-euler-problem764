//
// Created by jsier on 11/12/2021.
//

#include "problem764_cuda.cuh"
#include "../utils.cuh"
#include "../result.cuh"
#include "kernels.cuh"
#include "cuda_vector.cuh"

constexpr uint64_t calculate_number_blocks(uint64_t m_max) {
    const uint64_t sms = 40;
    const uint64_t max_threads_per_block = 1024;
    const uint64_t needed_treads = m_max;
    const uint64_t needed_blocks = ceil_div(needed_treads, max_threads_per_block);
    const uint64_t blocks_multiple_sms = ceil_div(needed_blocks, sms) * sms;


    // TODO, prevent blocks with less than 4 threads. This is just a hack for now
    const uint64_t threads_per_block = ceil_div(m_max, blocks_multiple_sms);
    if (m_max <= sms) {
        return 1;
    }
    if (threads_per_block < 4) {
        return blocks_multiple_sms / 4;
    }

    return blocks_multiple_sms;
}

void cuda_v0(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n) {
    CudaVector<uint64_t> outputs(blocks * threads_per_block, true);
    Result result("CUDA 1 LOOP", big_n);
    uint64_t sum_xyz = 0;

    result.cuda_tic("Kernel Launch", blocks * threads_per_block);
    kernel0<<<blocks, threads_per_block>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    result.set_sum_xyz(sum_xyz);
    result.print();
}

void cuda_v1(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n) {
    CudaVector<uint64_t> outputs(blocks * threads_per_block, true);
    Result result("CUDA 2 LOOP no opti", big_n);
    uint64_t sum_xyz = 0;

    result.cuda_tic("Kernel (A) Launch", blocks * threads_per_block);
    kernel1<<<blocks, threads_per_block>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    result.cuda_tic("Kernel (B) Launch", blocks * threads_per_block);
    kernel2<<<blocks, threads_per_block>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    result.set_sum_xyz(sum_xyz);
    result.print();
}

void cuda_v2(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n, const uint64_t m_max) {
    CudaVector<uint64_t> outputs(blocks * threads_per_block, true);
    Result result("CUDA 2 LOOP 1 opti", big_n);
    uint64_t sum_xyz = 0;

    result.cuda_tic("Kernel (A) Launch", blocks * threads_per_block / 2);
    kernel3<<<blocks, threads_per_block / 2>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    result.cuda_tic("Kernel (B) Launch", blocks * threads_per_block);
    kernel2<<<blocks, threads_per_block>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    result.set_sum_xyz(sum_xyz);
    result.print();
}

void cuda_v3(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n, const uint64_t m_max) {
    CudaVector<uint64_t> outputs(blocks * threads_per_block, true);
    Result result("CUDA 2 LOOP 2 opti", big_n);
    uint64_t sum_xyz = 0;

    result.cuda_tic("Kernel (A) Launch", blocks * threads_per_block / 2);
    kernel3<<<blocks, threads_per_block / 2>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    // Calculate blocks and threads for r and v
    const auto r_max = uint64_t(std::sqrt(m_max) + 1);
    const uint64_t r_blocks = calculate_number_blocks(r_max);
    const uint64_t r_threads_per_block = ceil_div(r_max, r_blocks);
    CudaVector<uint64_t> r_outputs(r_blocks * r_threads_per_block, true);

    result.cuda_tic("Kernel (B) Launch", r_blocks * r_threads_per_block);
    kernel4<<<r_blocks, r_threads_per_block>>>(r_outputs.get_device(), big_n, r_max, m_max);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    r_outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += r_outputs.accumulate_host_mod9();
    result.toc();

    result.set_sum_xyz(sum_xyz);
    result.print();
}

void cuda_v4(const uint64_t blocks, const uint64_t threads_per_block, const uint64_t big_n, const uint64_t m_max) {
    // Calculate blocks and threads for a and b
    const auto a_max = uint64_t(std::sqrt(m_max) + 1);
    const uint64_t a_blocks = calculate_number_blocks(a_max);
    const uint64_t a_threads_per_block = ceil_div(a_max, a_blocks);

    printf("%ld\n", a_max);
    CudaVector<uint64_t> outputs(a_blocks * a_threads_per_block, true);
    Result result("CUDA 2 LOOP 2 opti v2", big_n);
    uint64_t sum_xyz = 0;

    // When m is even
    result.cuda_tic("Kernel (A) Launch", a_blocks * a_threads_per_block);
    kernel5<<<a_blocks, a_threads_per_block>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

    // When m is odd
    result.cuda_tic("Kernel (A) Launch", a_blocks * a_threads_per_block);
    kernel6<<<a_blocks, a_threads_per_block>>>(outputs.get_device(), big_n);
    result.cuda_toc();

    result.cuda_tic("CPU -> GPU copy");
    outputs.device_to_host();
    result.cuda_toc();

    result.tic("Accumulate");
    sum_xyz += outputs.accumulate_host_mod9();
    result.toc();

//    // Calculate blocks and threads for r and v
//    const auto r_max = uint64_t(std::sqrt(m_max) + 1);
//    const uint64_t r_blocks = calculate_number_blocks(r_max);
//    const uint64_t r_threads_per_block = ceil_div(r_max, r_blocks);
//    CudaVector<uint64_t> r_outputs(r_blocks * r_threads_per_block, true);
//
//    result.cuda_tic("Kernel (B) Launch", r_blocks * r_threads_per_block);
//    kernel4<<<r_blocks, r_threads_per_block>>>(r_outputs.get_device(), big_n, r_max, m_max);
//    result.cuda_toc();
//
//    result.cuda_tic("CPU -> GPU copy");
//    r_outputs.device_to_host();
//    result.cuda_toc();
//
//    result.tic("Accumulate");
//    sum_xyz += r_outputs.accumulate_host_mod9();
//    result.toc();

    result.set_sum_xyz(sum_xyz);
    result.print();
}

void calculate764_cuda(const uint64_t big_n) {
    const auto m_max = uint64_t(std::sqrt(big_n) + 1);
    const uint64_t blocks = calculate_number_blocks(m_max);
    const uint64_t threads_per_block = ceil_div(m_max, blocks);
    const uint64_t total_threads = blocks * threads_per_block;

    printf("CUDA STATS:\n");
    printf(" * m := %ld iterations\n", m_max);
    printf(" * Blocks := %ld\n * Threads per block := %ld\n", threads_per_block, blocks);
    printf(" * Requires %.5f GB of VRAM\n", (double)(sizeof(uint64_t) * total_threads) / 1024 / 1024 / 1024);

    printf("\n");
    cuda_v0(blocks, threads_per_block, big_n);
    printf("\n");
    cuda_v1(blocks, threads_per_block, big_n);
    printf("\n");
    cuda_v2(blocks, threads_per_block, big_n, m_max);
    printf("\n");
    cuda_v3(blocks, threads_per_block, big_n, m_max);
    printf("\n");
    cuda_v4(blocks, threads_per_block, big_n, m_max);
    printf("\n");
}