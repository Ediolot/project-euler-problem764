//
// Created by jsier on 11/12/2021.
//

#include "utils.cuh"

uint64_t uint64_pow(uint64_t base, uint64_t exp) {
    uint64_t value = 1;
    while (exp > 0) {
        value *= base;
        exp--;
    }
    return value;
}

bool is_perfect_square(uint64_t n, uint64_t sqrt_n) {
    return n == sqrt_n * sqrt_n;
}

uint64_t gcd(uint64_t a, uint64_t b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}