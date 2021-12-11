//
// Created by jsier on 11/12/2021.
//


#include <cstdint>
#include <chrono>

#include "problem764_cpu.cuh"
#include "../utils.cuh"
#include "../result.cuh"


void update(uint64_t m, uint64_t big_n, uint64_t* n_elems, uint64_t* sum_xyz) {
    uint64_t step = m % 2 ? 2 : 1;
    uint64_t m2 = m * m;
    uint64_t mm = 2 * m;

    for (uint64_t n = step; n < m; n += step) {
        uint64_t n2 = n * n;
        uint64_t c = m2 + +n2;

        if (c > big_n) {
            break;
        }

        uint64_t a = m2 - n2;
        uint64_t b = mm * n;
        uint64_t y;

        y = (uint64_t) std::sqrt(b);
        if (!(a % 4) && is_perfect_square(b, y)) {
            uint64_t x = a / 4;
            if (gcd(x, y) == 1) {
                *n_elems += 1;
                *sum_xyz += x + y + c;
            }
        }

        y = (uint64_t) std::sqrt(a);
        if (!(b % 4) && is_perfect_square(a, y)) {
            uint64_t x = b / 4;
            if (gcd(x, y) == 1) {
                *n_elems += 1;
                *sum_xyz += x + y + c;
            }
        }
    }
}

void a() {
    printf("A");
}

void calculate764(uint64_t big_n) {
    Result result("CPU", big_n);

    uint64_t n_elems = 0;
    uint64_t sum_xyz = 0;
    auto m_max = uint64_t(std::sqrt(big_n) + 1);

    auto begin = std::chrono::steady_clock::now();
    result.tic("CPU Single thread");
    for (uint64_t m = 0; m < m_max; ++m) {
        update(m, big_n, &n_elems, &sum_xyz);
    }
    result.toc();

    result.set_sum_xyz(sum_xyz);
    result.print();
}