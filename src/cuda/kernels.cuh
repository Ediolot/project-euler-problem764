//
// Created by jsier on 09/12/2021.
//

#ifndef PROBLEM764_KERNELS_CUH
#define PROBLEM764_KERNELS_CUH

__device__ inline bool is_perfect_square_cuda(uint64_t n, uint64_t sqrt_n) {
    return n == sqrt_n * sqrt_n;
}

__device__ uint64_t gcd_cuda(uint64_t a, uint64_t b) {
    uint64_t aux;
    while (b > 0) {
        aux = a;
        a = b;
        b = aux % b;
    }
    return a;
}

// Single loop
__global__ void kernel0(uint64_t* __restrict__ outputs, uint64_t big_n) {
    uint64_t m = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum_xyz = 0;
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

        y = (uint64_t) sqrt((double)b);
        if (!(a % 4) && is_perfect_square_cuda(b, y)) {
            uint64_t x = a / 4;
            if (gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (c % 1000000000UL);
//                printf("A) %ld %ld %ld m=%ld n=%ld\n", x, y, c, m, n);
            }
        }

        y = (uint64_t) sqrt((double)a);
        if (!(b % 4) && is_perfect_square_cuda(a, y)) {
            uint64_t x = b / 4;
            if (gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (c % 1000000000UL);
//                printf("B) %ld %ld %ld m=%ld\n", x, y, c, m);
            }
        }
    }

    outputs[m] = sum_xyz;
}


// Two loops, no optims A
__global__ void kernel1(uint64_t* __restrict__ outputs, uint64_t big_n) {
    uint64_t m = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum_xyz = 0;
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

        y = (uint64_t) sqrt((double)b);
        if (!(a % 4) && is_perfect_square_cuda(b, y)) {
            uint64_t x = a / 4;
            if (gcd_cuda(x, y) == 1) {
//                printf("m=%ld, n=%ld step=%ld optim\n", m, n, step);
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (c % 1000000000UL);
//                printf("A) %ld %ld %ld\n", x, y, c);
            }
        }
    }

    outputs[m] = sum_xyz;
}

// Two loops, no optims B
__global__ void kernel2(uint64_t* __restrict__ outputs, uint64_t big_n) {
    uint64_t m = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum_xyz = 0;
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

        y = (uint64_t) sqrt((double)a);
        if (!(b % 4) && is_perfect_square_cuda(a, y)) {
            uint64_t x = b / 4;
            if (gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (c % 1000000000UL);
//                printf("%ld,%ld,%ld,%ld,%ld,%ld,%d\n", x, y, c, m, n, 0, 0);
            }
        }
    }

    outputs[m] = sum_xyz;
}

// Two loops, optim A
__global__ void kernel3(uint64_t* __restrict__ outputs, uint64_t big_n) {
    uint64_t m = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    uint64_t sum_xyz = 0;
    uint64_t m2 = m * m;
    uint64_t mm = 2 * m;

    uint64_t max_k = (m - (m % 4) - 2 + 4) / 4;  // +4 to prevent underflow (it's like adding +1 at the end)

    for (uint64_t k = 0; k < max_k; ++k) {
        uint64_t n = 2 + 4 * k + (m % 4);
        uint64_t n2 = n * n;
        uint64_t c = m2 + +n2;

        if (c > big_n) {
            break;
        }

        uint64_t a = m2 - n2;
        uint64_t b = mm * n;
        uint64_t y;

        y = (uint64_t) sqrt((double)b);
        if (!(a % 4) && is_perfect_square_cuda(b, y)) {
            uint64_t x = a / 4;
            if (gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (c % 1000000000UL);
//                printf("m=%ld, n=%ld opti, x=%ld, y=%ld, z=%ld\n", m, n, x, y, c);
//                sum_xyz += x + y + c;
//                printf("A) %ld %ld %ld\n", x, y, c);
            }
        }
    }

    outputs[m / 2] = sum_xyz;
}

// Two loops, optims B
__global__ void kernel4(uint64_t* __restrict__ outputs, uint64_t big_n, uint64_t r_max, uint64_t m_max) {
    uint64_t r = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum_xyz = 0;

//    const auto v_max = uint64_t(sqrtf((float)(m_max - r * r)) + 1);
    for (uint64_t v = 1; v < r; ++v) {
        uint64_t n = 2 * r * v;
        uint64_t m = r * r + v * v;

        uint64_t m2 = m * m;
        uint64_t mm = 2 * m;
        uint64_t n2 = n * n;
        uint64_t c = m2 + +n2;

        if (c > big_n) {
            break;
        }

        uint64_t a = m2 - n2;
        uint64_t b = mm * n;

        if (!(b % 4)) {  // We do not need to check that a is a perfect square, it already is
            uint64_t x = b / 4;
            auto y = (uint64_t) sqrt((double)a);
            if (gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (c % 1000000000UL);
//                sum_xyz += x + y + c;
//                printf("%ld,%ld,%ld,%ld,%ld,%ld,%d\n", x, y, c, m, n, v, r);
            }
        }
    }

    outputs[r] = sum_xyz;
}

// Two loops, fully optim A
__global__ void kernel5(uint64_t* __restrict__ outputs, uint64_t big_n) {
    uint64_t a = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum_xyz = 0;
    uint64_t m = 2 * a * a;
    uint64_t m2 = m * m;

    // TODO maxb = sqrt(2) * a
    uint64_t b_max = 2 * a;

    for (uint64_t b = 1; b < b_max; b += 2) { // B is odd
        uint64_t n = b * b;
        uint64_t n2 = n * n;
        uint64_t cc = 4 * (m2 + +n2);

        if (cc > big_n) {
            continue;
        }

        uint64_t aa = 4 * (m2 - n2);
        uint64_t bb = 8 * m * n;

        auto y = (uint64_t) sqrt((double)bb);
        if (!(aa % 4) && is_perfect_square_cuda(bb, y)) {
            uint64_t x = aa / 4;
            if ((16UL * x *  x + y * y * y * y - cc * cc) == 0 && gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (cc % 1000000000UL);
//                printf("m=%ld, n=%ld optimx2 a=%ld, b=%ld cc=%ld cc>bign=%d i\n", m, n, a, b, cc, cc > big_n);
//                sum_xyz += x + y + c;
//                printf("A) %ld %ld %ld\n", x, y, c);
            }
        }
    }

    outputs[a] = sum_xyz;
}

// Two loops, fully optim A
__global__ void kernel6(uint64_t* __restrict__ outputs, uint64_t big_n) {
    uint64_t a = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t sum_xyz = 0;
    uint64_t m = a * a;
    uint64_t m2 = m * m;

    // TODO maxb = a / sqrt(2)
    uint64_t b_max = a;

    if (a % 2 != 1) {
        return;
    }

    for (uint64_t b = 2; b < b_max; ++b) {
        uint64_t n = 2 * b * b;
        uint64_t n2 = n * n;
        uint64_t cc = 4 * (m2 + +n2);

        if (cc > big_n) {
            break;
        }

        uint64_t aa = 4 * (m2 - n2);
        uint64_t bb = 8 * m * n;

        auto y = (uint64_t) sqrt((double)bb);
        if (!(aa % 4) && is_perfect_square_cuda(b, y)) {
            uint64_t x = aa / 4;
            if (gcd_cuda(x, y) == 1) {
                sum_xyz += (x % 1000000000UL) + (y % 1000000000UL) + (cc % 1000000000UL);
//                printf("m=%ld, n=%ld optimx2 ii\n", m, n);
//                sum_xyz += x + y + c;
//                printf("A) %ld %ld %ld\n", x, y, c);
            }
        }
    }

    outputs[a] = sum_xyz;
}

#endif //PROBLEM764_KERNELS_CUH
