#pragma once 

#include"../Utils.hpp"
#include <cuda_runtime.h>



namespace GRS {


    __global__ void Upsweep(
        uint32_t* sort,
        uint32_t* globalHist,
        uint32_t* passHist,
        uint32_t size,
        uint32_t radixShift);

    __global__  void Scan(
        uint32_t* passHist,
        uint32_t  blocdimSize);

    __global__ void Sort(
        uint32_t* sort,
        uint32_t* result,//排序之后的数组
        uint32_t* globalHist,
        uint32_t* passHist,
        uint32_t sort_size,
        uint32_t radixShift
    );




    class CudaRadixSort {
        public:
            CudaRadixSort() {}
            ~CudaRadixSort() {} 
        

            void CudaSort(uint32_t* data, const uint32_t size);

            void CudaSort(float* data, const uint32_t size);

            //记录时间
            float elapsedTime = 0.0;

        private:
            //桶的大小，这里是256进制的桶
            const uint32_t k_radix = 256;
            //由于uint32_t有32位，而桶的大小是256，也就是8位
            //因此需要向GPU pass(提交) 32/8=4 回才能对整个数字排序完毕
            const uint32_t k_radixPasses = 4;

            const uint32_t m_SweepThreads = 128;

            const uint32_t m_SortThreads = 512;

            cudaEvent_t start, stop;
        
    };
}