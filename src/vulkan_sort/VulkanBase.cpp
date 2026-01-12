#include "VulkanBase.hpp"
#include<set>
namespace GRS {

    void VulkanBase::createInstance()
    {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Graffiti";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Graffiti Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo = { };
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        if (vkCreateInstance(&createInfo, nullptr, &m_Instance) != VK_SUCCESS) {
            GRS_ERROR("Failed to Create Instance, File:VulaknDevice.cpp");
        }

    }

    void VulkanBase::pickPhysicalDevice()
    {
        //两次调用vkEnumeratePhysicalDevices函数获取设备信息。
        //这个应该是c语言风格的代码，
        //第一次获取数量，第二次根据这个数量获取对应数量的信息
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            GRS_ERROR("Faild to find GPU with Vulkan support!  File:VulaknDevice.cpp");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                m_PhysicalDevice = device;
                break;
            }
        }
        if (m_PhysicalDevice == VK_NULL_HANDLE) {
            m_PhysicalDevice = devices[0];
            GRS_WARN("no suitalbe PhysicalDevice!");
        }
        vkGetPhysicalDeviceProperties(m_PhysicalDevice, &m_Properties);
        
    }

    void VulkanBase::createLogicalDevice()
    {
        //获取物理设备的队列索引

        QueueFamilyIndices indices = findQueueFamilies(m_PhysicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.computeFamily};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        //VkPhysicalDeviceProperties props{};
        //vkGetPhysicalDeviceProperties(m_PhysicalDevice, &props);
        //auto& limits = props.limits;
        //printf("maxComputeWorkGroupInvocations = %u\n",
        //    limits.maxComputeWorkGroupInvocations);
        //printf("maxComputeWorkGroupSize = %u %u %u\n",
        //    limits.maxComputeWorkGroupSize[0],
        //    limits.maxComputeWorkGroupSize[1],
        //    limits.maxComputeWorkGroupSize[2]);


        std::vector<std::string> unsupported;
        std::vector<const char*> enabledExtensions = FilterSupportedExtensions(m_PhysicalDevice, &unsupported);
        // 打印一下哪些扩展被跳过
        for (const auto& ext : unsupported) {
            GRS_WARN("Device does not support extension: {0}", ext);
        }
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = NULL;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();

        // might not really be necessary anymore because device specific validation layers
        // have been deprecated
    /*	if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }*/

        if (vkCreateDevice(m_PhysicalDevice, &createInfo, nullptr, &m_Device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(m_Device, indices.computeFamily, 0, &m_ComputeQueue);

    }
    void VulkanBase::createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findPhysicalQueueFamilies();

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily;
        poolInfo.flags =
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_CommandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void VulkanBase::createTimeStamp()
    {
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;   // start + end

        vkCreateQueryPool(m_Device, &qpci, nullptr, &m_QueryPool);

    }

    std::vector<const char*> VulkanBase::FilterSupportedExtensions(VkPhysicalDevice device, std::vector<std::string>* unsupportedExtensions)
    {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> availableSet;
        for (const auto& ext : availableExtensions) {
            availableSet.insert(ext.extensionName);
        }

        std::vector<const char*> enabledExtensions;
        for (const char* req : deviceExtensions) {
            if (availableSet.count(req)) {
                enabledExtensions.push_back(req); // 支持的加入
            }
            else if (unsupportedExtensions) {
                unsupportedExtensions->emplace_back(req); // 不支持的也记录一下（可选）
            }
        }

        return enabledExtensions;
    }


    bool VulkanBase::checkDeviceExtensionSupport(VkPhysicalDevice device)
    {

        uint32_t extensionCount;
        //依旧是用两次调用的方式去获取物理设备扩展的信息
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(
            device,
            nullptr,
            &extensionCount,
            availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        //很有意思的挑选方式，
        //如果物理设备里用我们需要的扩展就把对应扩展的名字擦除
        //如果看看最后有没有requiredExtensions空，要是空了说明都擦除完了，都有
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();

    }

    bool VulkanBase::isDeviceSuitable(VkPhysicalDevice device)
    {
        //查询物理设备的队列
        QueueFamilyIndices indices = findQueueFamilies(device);

        //判断物理设备是否支持需要的扩展
        bool extensionsSupported = checkDeviceExtensionSupport(device);


        //获取物理设备的特性
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        //我们需要的功能都支持的物理设备，就是我们需要的物理设备
        return indices.isComplete() && extensionsSupported  &&
            supportedFeatures.samplerAnisotropy;
    }

    QueueFamilyIndices VulkanBase::findQueueFamilies(VkPhysicalDevice device)
    {

        QueueFamilyIndices indices;

        //还是和查询物理设备的数量和信息一样的方式去查询物理设备的队列数量和信息
        uint32_t queueFamilyCount = 0;

        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        //遍历物理设备的每一个队列
        for (const auto& queueFamily : queueFamilies) {
            //判断这个队列是否大于0，可能是因为队列族里面有空队列？
            //判断这个队列是否支持计算队列

            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                indices.computeFamily = i;
                indices.computeFamilyHasValue = true;
            } 
            //如果已经有了我们想要的计算队列，那么我们就退出循环，就挑选好了队列
            if (indices.isComplete()) {
                break;
            }
            //要是没有选好，那就往下遍历
            i++;
        }
        //返回队列信息
        return indices;

    }

    uint32_t VulkanBase::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(m_PhysicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
    VkFormat VulkanBase::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(m_PhysicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (
                tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }


    void VulkanBase::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(m_Device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_Device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(m_Device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }
        vkBindBufferMemory(m_Device, buffer, bufferMemory, 0);


    }

    VkCommandBuffer VulkanBase::beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_CommandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(m_Device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        return commandBuffer;
    }
    void VulkanBase::endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(m_ComputeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_ComputeQueue);
        vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &commandBuffer);
    }

    void VulkanBase::TimeStart(VkCommandBuffer cmd) {
        vkCmdResetQueryPool(cmd, m_QueryPool, 0, 2);

        vkCmdWriteTimestamp(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            m_QueryPool, 0);
    }
    void  VulkanBase::TimeEnd(VkCommandBuffer cmd)
    {
        vkCmdWriteTimestamp(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            m_QueryPool, 1);
    }
    double VulkanBase::GetTime()
    {
        uint64_t timestamps[2];

        vkGetQueryPoolResults(
            m_Device,
            m_QueryPool,
            0, 2,
            sizeof(timestamps),
            timestamps,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        );
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_PhysicalDevice, &props);

        double period = props.limits.timestampPeriod;   // 纳秒 / tick

        return  (timestamps[1] - timestamps[0]) * period * 1e-6;
    }
}