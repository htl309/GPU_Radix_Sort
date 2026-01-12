#include "VulkanSort.hpp"

namespace GRS {

    VulkanRadixSort::VulkanRadixSort()
    {
        GRS_LOG("Vulkan Init")
        m_VulkanBase = std::make_shared<VulkanBase>();

        CreatePipeLineLayout();
        std::string  rootpath = "src\\vulkan_sort\\Shaders\\";
        m_floatcoderPipeLine = std::make_shared<VulkanPipeLine>(m_VulkanBase, rootpath+"FloatCoder.comp.spv", m_PipelineLayout);
        m_upsweepPipeLine = std::make_shared<VulkanPipeLine>(m_VulkanBase, rootpath+"UpSweep.comp.spv", m_PipelineLayout);
        m_scanPipeLine = std::make_shared<VulkanPipeLine>(m_VulkanBase, rootpath+"Scan.comp.spv", m_PipelineLayout);
        m_sortPipeLine = std::make_shared<VulkanPipeLine>(m_VulkanBase, rootpath+"Sort.comp.spv", m_PipelineLayout);

        m_Barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        m_Barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        m_Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    }

    void VulkanRadixSort::VulkanSort(uint32_t* data, uint32_t size) {

        CreateBuffer(size);
        m_BufferSort->UploadKeys((void*)data, size);

        const  uint32_t BlockDimsize = divRoundUp(size, PART_SIZE);

        VkCommandBuffer cmd = m_VulkanBase->beginSingleTimeCommands();
        m_VulkanBase->TimeStart(cmd);
        m_VulkanBase->GetTimePool();
        m_PushconstData.sort_size = size;
        m_PushconstData.BlockDimSize = BlockDimsize;

        for (uint32_t i = 0; i < k_radixPasses; i++) {

            m_PushconstData.radixShift = i * 8;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upsweepPipeLine->GetPipeLine());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
            vkCmdDispatch(cmd, BlockDimsize, 1, 1);


            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &m_Barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_scanPipeLine->GetPipeLine());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
            vkCmdDispatch(cmd, RADIX, 1, 1);


            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &m_Barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sortPipeLine->GetPipeLine());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
            vkCmdDispatch(cmd, BlockDimsize, 1, 1);

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &m_Barrier, 0, nullptr, 0, nullptr);
            if (i < k_radixPasses - 1) SwapIO();

        }

        m_VulkanBase->TimeEnd(cmd);

        m_VulkanBase->endSingleTimeCommands(cmd);


        m_BufferResult->DownloadResult((void*)data, size);

        elapsedTime = m_VulkanBase->GetTime();

    }

    void VulkanRadixSort::VulkanSort(float* data, uint32_t size) {

        CreateBuffer(size);
        m_BufferSort->UploadKeys((void*)data, size);

        const  uint32_t BlockDimsize = divRoundUp(size, PART_SIZE);

        VkCommandBuffer cmd = m_VulkanBase->beginSingleTimeCommands();
        m_VulkanBase->TimeStart(cmd);
        m_VulkanBase->GetTimePool();
        m_PushconstData.sort_size = size;
        m_PushconstData.BlockDimSize = BlockDimsize;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_floatcoderPipeLine->GetPipeLine());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
        vkCmdDispatch(cmd, BlockDimsize, 1, 1);


        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &m_Barrier, 0, nullptr, 0, nullptr);

        for (uint32_t i = 0; i < k_radixPasses; i++) {

            m_PushconstData.radixShift = i * 8;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upsweepPipeLine->GetPipeLine());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
            vkCmdDispatch(cmd, BlockDimsize, 1, 1);


            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &m_Barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_scanPipeLine->GetPipeLine());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
            vkCmdDispatch(cmd, RADIX, 1, 1);


            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &m_Barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sortPipeLine->GetPipeLine());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
            vkCmdDispatch(cmd, BlockDimsize, 1, 1);

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &m_Barrier, 0, nullptr, 0, nullptr);
            SwapIO();

        }
        m_PushconstData.type = 1;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_floatcoderPipeLine->GetPipeLine());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushconstData), &m_PushconstData);
        vkCmdDispatch(cmd, BlockDimsize, 1, 1);

        m_VulkanBase->TimeEnd(cmd);

        m_VulkanBase->endSingleTimeCommands(cmd);


        m_BufferResult->DownloadResult((void*)data, size);

        elapsedTime = m_VulkanBase->GetTime();

    }


    void VulkanRadixSort::CreateBuffer(uint32_t size)
    {
        

        auto usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        auto props =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        const uint32_t blockdimsize = divRoundUp(size, PART_SIZE);

        VkDeviceSize SortSize = blockdimsize * PART_SIZE * UINT32_T_SIZE;
        VkDeviceSize GlobalHistSize = 256 * k_radixPasses * UINT32_T_SIZE;
        VkDeviceSize PassHistSize = blockdimsize * 256 * UINT32_T_SIZE;

        m_BufferSort = std::make_shared<VulkanBuffer>(m_VulkanBase, SortSize, usage, props);
        m_BufferResult = std::make_shared<VulkanBuffer>(m_VulkanBase, SortSize, usage, props);
        m_BufferGlobalHist = std::make_shared<VulkanBuffer>(m_VulkanBase, GlobalHistSize, usage, props);
        m_BufferPassHist = std::make_shared<VulkanBuffer>(m_VulkanBase, PassHistSize, usage, props);

       VkDescriptorBufferInfo bufA{ m_BufferSort->m_Buffer, 0, VK_WHOLE_SIZE };
       VkDescriptorBufferInfo bufB{ m_BufferResult->m_Buffer, 0, VK_WHOLE_SIZE };
       VkDescriptorBufferInfo bufC{ m_BufferGlobalHist->m_Buffer, 0, VK_WHOLE_SIZE };
       VkDescriptorBufferInfo bufD{ m_BufferPassHist->m_Buffer, 0, VK_WHOLE_SIZE };

       std::vector<VkWriteDescriptorSet> writes(4);

       auto write = [&](int i, VkDescriptorBufferInfo& info)
       {
           writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
           writes[i].dstSet = m_DescSet;
           writes[i].dstBinding = i;          // binding = 0,1,2,3
           writes[i].dstArrayElement = 0;
           writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
           writes[i].descriptorCount = 1;
           writes[i].pBufferInfo = &info;
       };

       write(0, bufA);
       write(1, bufB);
       write(2, bufC);
       write(3, bufD);

       vkUpdateDescriptorSets(m_VulkanBase->GetVulkanDevice(), 4, writes.data(), 0, nullptr);

    }




    void VulkanRadixSort::CreatePipeLineLayout()
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(4);

        for (uint32_t i = 0; i < 4; i++)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(m_VulkanBase->GetVulkanDevice(), &layoutInfo, nullptr, &m_DescLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }


        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 100;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        vkCreateDescriptorPool(m_VulkanBase->GetVulkanDevice(), &poolInfo, nullptr, &m_DescPool);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_DescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_DescLayout;

        vkAllocateDescriptorSets(m_VulkanBase->GetVulkanDevice(), &allocInfo, &m_DescSet);

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushconstData);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount =1;
        pipelineLayoutInfo.pSetLayouts = &m_DescLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        if (vkCreatePipelineLayout(m_VulkanBase->GetVulkanDevice(), &pipelineLayoutInfo, nullptr, &m_PipelineLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    void VulkanRadixSort::SwapIO()
    {
        std::swap(m_BufferSort, m_BufferResult);
        

        VkDescriptorBufferInfo bufA{ m_BufferSort->m_Buffer,  0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo bufB{ m_BufferResult->m_Buffer,0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[2]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_DescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufA;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_DescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &bufB;

        vkUpdateDescriptorSets(m_VulkanBase->GetVulkanDevice(), 2, writes, 0, nullptr);
    }






}
