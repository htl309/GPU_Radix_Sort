#include "VulkanBuffer.hpp"

namespace GRS {

    VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanBase> vulkanbase, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
        :m_VulkanBase(vulkanbase)
    {
        m_VulkanBase->createBuffer(size, usage, properties, m_Buffer, m_Memory);
   
    }

    void VulkanBuffer::UploadKeys(void* data, uint32_t size)
    {
        VkDeviceSize bytes = size * UINT32_T_SIZE;

        void* mapped;
        vkMapMemory(m_VulkanBase->GetVulkanDevice(), m_Memory, 0, bytes, 0, &mapped);
        memcpy(mapped, data, bytes);
        vkUnmapMemory(m_VulkanBase->GetVulkanDevice(), m_Memory);
    }

    void VulkanBuffer::DownloadResult(void* data, uint32_t size)
    {
        VkDeviceSize bytes = size * UINT32_T_SIZE;

        void* mapped;
        vkMapMemory(m_VulkanBase->GetVulkanDevice(), m_Memory, 0, bytes, 0, &mapped);
        memcpy(data, mapped, bytes);
        vkUnmapMemory(m_VulkanBase->GetVulkanDevice(), m_Memory);
    }

}
