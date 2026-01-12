glslc --target-env=vulkan1.2 FloatCoder.comp  -o FloatCoder.comp.spv

glslc --target-env=vulkan1.2 UpSweep.comp  -o UpSweep.comp.spv
glslc --target-env=vulkan1.2 Scan.comp  -o Scan.comp.spv
glslc --target-env=vulkan1.2 Sort.comp  -o Sort.comp.spv

pause