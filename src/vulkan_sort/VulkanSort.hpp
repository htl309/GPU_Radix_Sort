#include "VulkanBase.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanBuffer.hpp"
namespace GRS {
    class VulkanRadixSort {
        public:
            VulkanRadixSort();
            ~VulkanRadixSort() {}

            void   VulkanSort(uint32_t* data, uint32_t size);
            void   VulkanSort(float* data, uint32_t size);

            float elapsedTime = 0.0;
        private:
            void CreateBuffer(uint32_t size);
            void CreatePipeLineLayout();

            void SwapIO();

            VkMemoryBarrier m_Barrier{};
        private:
            std::shared_ptr<VulkanBase> m_VulkanBase;

            std::shared_ptr < VulkanPipeLine> m_floatcoderPipeLine;
            std::shared_ptr < VulkanPipeLine> m_upsweepPipeLine;
            std::shared_ptr < VulkanPipeLine> m_scanPipeLine;
            std::shared_ptr < VulkanPipeLine> m_sortPipeLine;

            VkPipelineLayout m_PipelineLayout;

            VkDescriptorSetLayout m_DescLayout;
            VkDescriptorPool      m_DescPool;
            VkDescriptorSet       m_DescSet;

            std::shared_ptr < VulkanBuffer > m_BufferSort;
            std::shared_ptr < VulkanBuffer > m_BufferResult;
            std::shared_ptr < VulkanBuffer > m_BufferGlobalHist;
            std::shared_ptr < VulkanBuffer > m_BufferPassHist;

            const uint32_t k_radixPasses = 4;
            struct PushconstData {
                uint32_t sort_size = 0;
                uint32_t radixShift = 0;
                uint32_t  BlockDimSize=0;
                uint32_t  type=0;
            } m_PushconstData;
           
            

    };

}
