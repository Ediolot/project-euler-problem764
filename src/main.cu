#include <iostream>

#include "cpu/problem764_cpu.cuh"
#include "cuda/problem764_cuda.cuh"
#include "utils.cuh"


int main() {
    auto big_n = uint64_pow(10, 7);
    calculate764_cuda(big_n);
    calculate764(big_n);
    return 0;
}
