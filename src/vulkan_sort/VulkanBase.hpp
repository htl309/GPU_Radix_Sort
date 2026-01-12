#pragma once
#include<vulkan/vulkan.h>
#include<memory>
#include "Utils.hpp"
namespace GRS{
    //队列索引
//我们调好的队列的索引就放在这里面
    struct QueueFamilyIndices {

        //我们只需要一个队列就行了，支持computeshader
        uint32_t computeFamily;

        bool computeFamilyHasValue = false;
        bool isComplete() { return computeFamilyHasValue; }
    };

    class VulkanBase{
        public:
        VulkanBase() {
            init();
        }
        inline VkDevice GetVulkanDevice() { return m_Device; }
        inline VkQueryPool  GetTimePool() { return m_QueryPool; }
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(m_PhysicalDevice); }
        VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);

        void TimeStart(VkCommandBuffer cmd);
        void TimeEnd(VkCommandBuffer cmd);
        double GetTime();
    private:
            void init() {

                createInstance();
                pickPhysicalDevice();
                createLogicalDevice();
                createCommandPool();
                createTimeStamp();
            }

            //init function
            void createInstance();
            void pickPhysicalDevice();
            void createLogicalDevice();
            void createCommandPool();
            void createTimeStamp();
            std::vector<const char*> FilterSupportedExtensions(VkPhysicalDevice device, std::vector<std::string>* unsupportedExtensions);



            bool isDeviceSuitable(VkPhysicalDevice device);
            QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
            bool checkDeviceExtensionSupport(VkPhysicalDevice device);


        private:

            VkInstance m_Instance;

            VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
            VkPhysicalDeviceProperties m_Properties;

            VkCommandPool m_CommandPool;

            VkDevice m_Device = VK_NULL_HANDLE;

            VkQueue m_ComputeQueue;

            const std::vector<const char*> deviceExtensions = {};

        private:
            VkQueryPool m_QueryPool;




            
    };


}


