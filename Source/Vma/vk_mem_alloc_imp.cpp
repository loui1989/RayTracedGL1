#ifdef __APPLE__
// macOS: Vulkan is implemented on Metal (MetalNative.h), there's no Vulkan loader to link
// against; MemoryAllocator gives VMA the native functions
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"