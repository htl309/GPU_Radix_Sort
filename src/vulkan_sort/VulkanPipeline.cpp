#include "VulkanPipeline.hpp"
#include<fstream>
namespace GRS {

    VulkanPipeLine::VulkanPipeLine(std::shared_ptr<VulkanBase> vulkanbase, std::string shaderfile, VkPipelineLayout pipelinelayout)
        :m_VulkanBase(vulkanbase)
    {

        LoadShader(shaderfile);

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelinelayout;
        pipelineInfo.stage = m_ShaderStage;

        if (vkCreateComputePipelines(
            m_VulkanBase->GetVulkanDevice(),
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &m_Pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline");
        }
    }
    void VulkanPipeLine::LoadShader(std::string shaderfile)
    {

        std::ifstream file{ shaderfile, std::ios::ate | std::ios::binary };

        if(!file.is_open())
        GRS_LOG( "Can't open ShaderFile!!! \n  Please run compile.bat File in Shaders!!! (Double click .bat)");

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = buffer.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

        if (vkCreateShaderModule(m_VulkanBase->GetVulkanDevice(), &createInfo, nullptr, &m_ShaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module");
        }

        m_ShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        m_ShaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        m_ShaderStage.module = m_ShaderModule;
        m_ShaderStage.pName = "main";
        m_ShaderStage.flags = 0;
        m_ShaderStage.pNext = nullptr;
        m_ShaderStage.pSpecializationInfo = nullptr;

    }
}

