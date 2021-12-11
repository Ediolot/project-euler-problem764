//
// Created by jsier on 11/12/2021.
//

#include <string>
#include <iostream>
#include <sstream>

#include "result.cuh"
#include "utils.cuh"


Result::Result(std::string name, uint64_t problem_size)
    : name(std::move(name))
    , problem_size(problem_size)
{}

void Result::tic(std::string key) {
    current_key = std::move(key);
    begin = std::chrono::steady_clock::now();
}

void Result::toc() {
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    timings.emplace_back(current_key, (double)us / 1000.0);
    current_key = "";
}

void Result::cuda_tic(std::string key, uint64_t total_threads) {
    current_key = std::move(key);

    if (total_threads > 0) {
        // TODO, clean up
        char buffer[512];
        sprintf(buffer, " [%ld threads]", total_threads);
        current_key += std::string(buffer);
    }

    cudaEventCreate(&cuda_begin);
    cudaEventRecord(cuda_begin);
}

void Result::cuda_toc() {
    float ms;
    cudaEventCreate(&cuda_end);
    cudaEventRecord(cuda_end);
    cudaEventSynchronize(cuda_end);
    cudaEventElapsedTime(&ms, cuda_begin, cuda_end);
    timings.emplace_back(current_key, (double)ms);
    current_key = "";
}

void Result::set_sum_xyz(uint64_t value) {
    sum_xyz = value;
}

void Result::print() const {
    double total_us = 0;
    std::stringstream time_detail;
    time_detail.precision(3);

    for (const auto& element : timings) {
        auto section = element.first;
        auto time_us = element.second;
        time_detail << section << ": " << std::fixed << time_us << " ms | ";
        total_us += time_us;
    }

    printf("[%-20s] Took %.3f ms. S(%.0e) = %ld (%ld mod 1e9).\n     -> %s\n",
           name.c_str(), total_us, (double)problem_size, sum_xyz, sum_xyz % uint64_pow(10, 9), time_detail.str().c_str());
}
