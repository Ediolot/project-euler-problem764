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

uint64_t gcd(uint64_t a, uint64_t b) {
    uint64_t aux;
    while (b > 0) {
        aux = a;
        a = b;
        b = aux % b;
    }
    return a;
}