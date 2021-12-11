#!/usr/bin/python3

from math import gcd, sqrt
from multiprocessing import Pool
from time import perf_counter as pc


def Calcula(m, N):
    nelem = 0
    suma = 0
    m2 = m * m
    mm = 2 * m
    ## nmax = int(sqrt(N - m * m))
    # m ó n han de ser pares
    start_step = 2 if m % 2 else 1
    for n in range(start_step, m, start_step):
        # if n > nmax:
        #     break
        if (c := m2 + (n2 := n * n)) > N:
            break
        a = m2 - n2
        b = mm * n      # 2 * m * n
        # 4x = a, y^2 = b, z = c
        if not a % 4 and not (y := sqrt(b)) % 1:
            x = a // 4
            y = int(y)
            if gcd(x, y) == 1:
                nelem += 1
                suma += x + int(y) + c
                # print(f'{m:2} {n:2}  ({x:3} {int(y):3} {c:3})')
        # y^2 = a, 4x = b, z = c
        if not b % 4 and not (y := sqrt(a)) % 1:
            x = b // 4
            y = int(y)
            if gcd(x, y) == 1:
                nelem += 1
                suma += x + int(y) + c
                # print(f'{m:2} {n:2}  ({x:3} {int(y):3} {c:3})')
    return nelem, suma


def CB_Calcula(result):
    global solucion

    solucion[0] += result[0]
    solucion[1] += result[1]
    return


def main():
    global solucion

    MP_ENABLED = 0

    N = 100
    N = 10**16
    N = 10**4
    N = 10**7

    solucion = [0, 0]

    if MP_ENABLED:
        pool = Pool()   # Por defecto usa cpu_count()

    mmax = int(sqrt(N))
    # print(f'Max m: {mmax}, current: 0\r', end='')
    msg_time = pc() - 99
    for m in range(mmax + 1, 0, -1):
        if pc() - msg_time > 10:
            msg_time = pc()
            print(f'\r Iterations left: {m:,} ', end='')
        if MP_ENABLED:
            pool.apply_async(Calcula, (m, N), callback=CB_Calcula)
        else:
            nelem, suma = Calcula(m, N)
            solucion[0] += nelem
            solucion[1] += suma
    if MP_ENABLED:
        pool.close()
        pool.join()
    print('\r', ' ' * 55)

    nelem = solucion[0]
    suma = solucion[1]
    print(f'S({N:2.0e}): {suma} ({suma % 10**9} mod 10^9), with {nelem} solutions')


if __name__ == '__main__':
    time_start = pc()
    main()
    time_end = pc()
    print(f'\nElapsed time: {time_end - time_start:.2f} s')
