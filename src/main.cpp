#include<iostream>
#include"vulkan_sort/VulkanSort.hpp"
#include"cuda_sort/cudasort_h.cu"

using namespace std;

template <typename T>
void CudaTest() {
    GRS_LOG("---------------------cudaSort start----------------------------")
    std::cout <<"DataType(uint32_t/float) : "<< typeid(T).name() << std::endl;
    GRS::CudaRadixSort cudasort;
    GRS_LOG("--------cudaSort Correctness test----------------------------")

        for (uint32_t i = 1 << 4; i < (1 << 26); i = i << 3) {
            uint32_t size = i;
            std::vector<T>  data(size);
            createRandomArray(data.data(), size);
            cudasort.CudaSort(data.data(), size);
            if (isAscending(data.data(), size))
                std::cout << "\033[1;32mCudaSort | ArrayLength:" << size << ", Correct! \033[0m" << std::endl;

            data.clear();
        }

    GRS_LOG("------cudaSort Time test----------------------------")
    float time = 0.0;
    uint32_t size = 1 << 28;

    for (uint32_t i = 0; i < 5; i++) {

        std::vector<T>  data(size);
        createRandomArray(data.data(), size);

        cudasort.CudaSort(data.data(), size);
        time += cudasort.elapsedTime;
        cout << cudasort.elapsedTime << "ms ";
        data.clear();

    }

    std::cout << std::endl << "\033[1;32mCudaSort | ArrayLength:" << size << " (2^28), Time:" << time / 5.0 << "ms \033[0m" << std::endl;
    GRS_LOG("---------------------cudaSort end----------------------------")
    std::cout << endl;
}


template <typename T>
void VulkanTest() {
    GRS_LOG("---------------------vulkanSort start----------------------------")
        std::cout << "DataType(uint32_t/float) : " << typeid(T).name() << std::endl;
    GRS::VulkanRadixSort vulkanSort;
    GRS_LOG("--------vulkanSort Correctness test----------------------------")

        for (uint32_t i = 1 << 4; i < (1 << 26); i = i << 3) {
            uint32_t size = i;
            std::vector<T>  data(size);
            createRandomArray(data.data(), size);
            vulkanSort.VulkanSort(data.data(), size);
            if (isAscending(data.data(), size))
                std::cout << "\033[1;32m vulkanSort | ArrayLength:" << size << ", Correct! \033[0m" << std::endl;

            data.clear();
        }

    GRS_LOG("------vulkanSort Time test----------------------------")
        float time = 0.0;
    uint32_t size = 1 << 28;

    for (uint32_t i = 0; i < 5; i++) {

        std::vector<T>  data(size);
        createRandomArray(data.data(), size);

        vulkanSort.VulkanSort(data.data(), size);
        time += vulkanSort.elapsedTime;
        cout << vulkanSort.elapsedTime << "ms ";
        data.clear();

    }

    std::cout << std::endl << "\033[1;32m vulkanSort | ArrayLength:" << size << " (2^28), Time:" << time / 5.0 << "ms \033[0m" << std::endl;
    GRS_LOG("---------------------vulkanSort end----------------------------")
        std::cout << endl;
}

int main() {

    CudaTest<uint32_t>();

    //CudaTest<float>();
    //VulkanTest<uint32_t>();
    //VulkanTest<float>();

    return 0;
}