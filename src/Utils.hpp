#pragma once
#include <iostream>
#include <random>
#include <filesystem>  // C++17+
#include"define.h"
#define GRS_LOG(msg) \
    std::cout << "\033[1;32m[GPUSORT LOG] " << std::filesystem::path(__FILE__).filename() << " LINE:" << __LINE__ << " | " << msg  << "\033[0m"<< std::endl;

#define GRS_WARN(msg) \
std::cout << "\033[1;33m[GPUSORT WARN] " <<  std::filesystem::path(__FILE__).filename() <<" LINE:"<<__LINE__<<" | " << msg << "\033[0m" << std::endl;

#define GRS_ERROR(msg)  \
std::cerr << "\033[1;31m[GPUSORT ERROR] " <<  std::filesystem::path(__FILE__).filename() <<" LINE:"<<__LINE__<<" | " << msg << "\033[0m" << std::endl;
 
#define GRS_CUDA_LOG(msg) \
printf("\033[1;32m  [GPUSORT CUDALOG] | ");printf(msg);printf(" \033[0m");

#define ARRAY_LOG(data,size) do{\
for (uint32_t i = 0; i < size; i++)  \
    std::cout << data[i] << " "; \
std::cout << std::endl;\
}while(false);

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        printf("CUDA error at %s:%d: %s\n",                   \
               __FILE__, __LINE__,                             \
               cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                   \
    }                                              \
} while (false)

inline uint32_t divRoundUp(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}

inline void createRandomArray(uint32_t * data, uint32_t datasize, uint32_t maxValue = UINT32_MAX) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(
        std::numeric_limits<uint32_t>::min(),
        maxValue
    );    
    for (int i = 0; i < datasize; ++i) {
        data[i] = dist(gen); 
    }
}

inline void createRandomArray(float* data, uint32_t datasize, float maxValue = 1<<25) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(
       -maxValue,
        maxValue
    );
    for (int i = 0; i < datasize; ++i) {
        data[i] =  dist(gen);
    }
}

inline bool isAscending(uint32_t* arr, uint32_t size) {
    // 遍历数组，比较相邻元素
    for (uint32_t i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            GRS_WARN("Wrong sorting");
            GRS_WARN(i);
            return false;  // 如果发现某个元素大于下一个元素，则不是升序数组
        }
    }
    return true;  // 如果所有元素都满足升序条件，返回true
}
inline bool isAscending(float* arr, uint32_t size) {
    // 遍历数组，比较相邻元素
    for (uint32_t i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            GRS_WARN("Wrong sorting");
            GRS_WARN(arr[i]);
            return false;  // 如果发现某个元素大于下一个元素，则不是升序数组
        }
    }
    return true;  // 如果所有元素都满足升序条件，返回true
}