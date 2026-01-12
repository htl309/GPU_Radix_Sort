#include "VulkanBase.hpp"


namespace GRS {
    class VulkanBuffer{

        public:
            VulkanBuffer(std::shared_ptr<VulkanBase> vulkanbase, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
          
            ~VulkanBuffer() {
             
                if (m_Buffer != VK_NULL_HANDLE) {
                    vkDestroyBuffer(m_VulkanBase->GetVulkanDevice(), m_Buffer, nullptr);
                }
                if (m_Memory != VK_NULL_HANDLE) {
                    vkFreeMemory(m_VulkanBase->GetVulkanDevice(), m_Memory, nullptr);
                }
                
            }

            void UploadKeys(void* data, uint32_t size);
            void DownloadResult(void* data, uint32_t size);
          
            
        
            VkBuffer m_Buffer;
            VkDeviceMemory m_Memory;
            private:
            std::shared_ptr<VulkanBase> m_VulkanBase;
    };

}
