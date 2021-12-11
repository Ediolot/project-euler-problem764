//
// Created by jsier on 11/12/2021.
//

#ifndef PROBLEM764_UTILS_CUH
#define PROBLEM764_UTILS_CUH


#include <cstdint>

uint64_t uint64_pow(uint64_t base, uint64_t exp);
bool is_perfect_square(uint64_t n, uint64_t sqrt_n);
uint64_t gcd(uint64_t a, uint64_t b);
constexpr uint64_t ceil_div(uint64_t x, uint64_t y) {
    return x / y + (x % y != 0);
}


#endif //PROBLEM764_UTILS_CUH
