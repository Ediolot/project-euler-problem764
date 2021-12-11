//
// Created by jsier on 11/12/2021.
//

#ifndef PROBLEM764_RESULT_CUH
#define PROBLEM764_RESULT_CUH

#include <vector>
#include <chrono>
#include <iostream>

class Result {
public:
    Result(std::string name, uint64_t problem_size);

    void tic(std::string key);
    void toc();
    void cuda_tic(std::string key, uint64_t total_threads = 0);
    void cuda_toc();

    void set_sum_xyz(uint64_t value);

    void print() const;

private:
    std::vector<std::pair<std::string, double>> timings{};  // key, ms
    uint64_t sum_xyz = 0;
    uint64_t n_elems = 0;

    std::chrono::steady_clock::time_point begin{};
    cudaEvent_t cuda_begin{}, cuda_end{};
    std::string current_key{};
    std::string name{};
    uint64_t problem_size = 0;

};


#endif //PROBLEM764_RESULT_CUH
