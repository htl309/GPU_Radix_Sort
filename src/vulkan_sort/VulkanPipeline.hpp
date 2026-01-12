#include "VulkanBase.hpp"

namespace GRS {


    class VulkanPipeLine {
        public:
            VulkanPipeLine(std::shared_ptr<VulkanBase> vulkanbase, std::string shaderfile, VkPipelineLayout pipelinelayout);
            ~VulkanPipeLine() {}

            inline VkPipeline  GetPipeLine() { return m_Pipeline; }
        private:

            void LoadShader(std::string shaderfile);

            VkPipeline m_Pipeline = VK_NULL_HANDLE;
            VkPipelineShaderStageCreateInfo m_ShaderStage;
            VkShaderModule m_ShaderModule;

            std::shared_ptr<VulkanBase> m_VulkanBase;
    };

}
