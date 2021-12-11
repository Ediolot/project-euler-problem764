//
// Created by jsier on 11/12/2021.
//

#ifndef PROBLEM764_CUDAVECTOR_CUH
#define PROBLEM764_CUDAVECTOR_CUH

template<class T>
class CudaVector {
public:
    CudaVector() = default;

    explicit CudaVector(size_t size, bool initialize = false)
        : size(size)
    {
        cudaMalloc((void**)&device, sizeof(T) * size);

        if (initialize) {
            host = (T*)calloc(size, sizeof(T));
            host_to_device();
        } else {
            host = (T*)malloc(sizeof(T) * size);
        }
    }

    ~CudaVector() {
        free(host);
        cudaFree(device);
    }

    T* device_to_host() {
        cudaMemcpy(host, device, sizeof(T) * size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return host;
    }

    T* host_to_device() {
        cudaMemcpy(device, host, sizeof(T) * size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        return device;
    }

    T* get_host() {
        return host;
    }

    T* get_device() {
        return device;
    }

    size_t get_size() const {
        return size;
    }

    T accumulate_host() {
        T acc = 0;
        for (size_t i = 0; i < size; ++i) {
            acc += host[i];
        }
        return acc;
    }

    T accumulate_host_mod9() {
        T acc = 0;
        for (size_t i = 0; i < size; ++i) {
            acc += host[i] % 1000000000UL;
        }
        return acc;
    }

private:
    T* host = nullptr;
    T* device = nullptr;
    size_t size = 0;
};

#endif //PROBLEM764_CUDAVECTOR_CUH
