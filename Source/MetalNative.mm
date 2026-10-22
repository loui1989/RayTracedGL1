// Copyright (c) 2026 Quake RT for macOS contributors
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// macOS, RTGL1_METAL_ONLY=1: the Vulkan functions RTGL1 uses, implemented on Metal.
// Objects (instance, devices, memory, buffers, images, sync, swapchain, ...) are here;
// command buffers and queue submission are in MetalBridge.mm, where the commands are
// encoded with the same code as the native sections. Only what RTGL1 needs is supported.

#define RG_METAL_BRIDGE_IMPLEMENTATION // the calls in this file aren't routed by MetalShims.h
#include "MetalNativeInternal.h"
#include "MetalBridge.h"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>

using namespace RTGL1::MetalNative;

namespace
{

void Log( const char* msg )
{
    fprintf( stderr, "[Metal-only] %s\n", msg );
}

template< typename T >
T* Obj( uint64_t handle )
{
    return reinterpret_cast< T* >( uintptr_t( handle ) );
}
template< typename T, typename H >
T* Obj( H* handle )
{
    return reinterpret_cast< T* >( handle );
}
template< typename H, typename T >
H Handle( T* obj )
{
    if constexpr( std::is_pointer_v< H > )
    {
        return reinterpret_cast< H >( obj );
    }
    else
    {
        return H( uintptr_t( obj ) );
    }
}

// objects that need nothing but a unique handle
struct Dummy
{
    uint64_t unused = 0;
};

struct NativeDescriptorPool
{
    std::vector< Dummy* > sets;
};

uint64_t g_instance, g_physicalDevice, g_device, g_queue; // their addresses are the handles

struct Globals
{
    id< MTLDevice >       device;
    id< MTLCommandQueue > queue;

    std::mutex                               pendingLock;
    std::vector< id< MTLCommandBuffer > >    pending; // committed, maybe not completed

    std::atomic< uint64_t > nextASAddress{ 0x7f0000000000ull };
};

Globals& G()
{
    static Globals g;
    return g;
}

template< typename T >
VkResult FillArray( uint32_t* count, T* out, const T* src, uint32_t srcCount )
{
    if( !out )
    {
        *count = srcCount;
        return VK_SUCCESS;
    }
    const uint32_t n = std::min( *count, srcCount );
    for( uint32_t i = 0; i < n; i++ )
    {
        out[ i ] = src[ i ];
    }
    *count = n;
    return n < srcCount ? VK_INCOMPLETE : VK_SUCCESS;
}

VkExtensionProperties Ext( const char* name, uint32_t version = 1 )
{
    VkExtensionProperties p = {};
    strncpy( p.extensionName, name, VK_MAX_EXTENSION_NAME_SIZE - 1 );
    p.specVersion = version;
    return p;
}

// every VkBool32 of a feature structure (they have nothing else after sType / pNext)
void EnableAllFeatures( void* s, size_t size )
{
    auto* b = reinterpret_cast< uint8_t* >( s ) + offsetof( VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge );
    auto* e = reinterpret_cast< uint8_t* >( s ) + size;
    for( ; b + sizeof( VkBool32 ) <= e; b += sizeof( VkBool32 ) )
    {
        *reinterpret_cast< VkBool32* >( b ) = VK_TRUE;
    }
}

MTLTextureType TextureTypeOf( VkImageViewType t, bool& ok )
{
    ok = true;
    switch( t )
    {
        case VK_IMAGE_VIEW_TYPE_1D: return MTLTextureType1D;
        case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return MTLTextureType1DArray;
        case VK_IMAGE_VIEW_TYPE_2D: return MTLTextureType2D;
        case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return MTLTextureType2DArray;
        case VK_IMAGE_VIEW_TYPE_CUBE: return MTLTextureTypeCube;
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return MTLTextureTypeCubeArray;
        case VK_IMAGE_VIEW_TYPE_3D: return MTLTextureType3D;
        default: ok = false; return MTLTextureType2D;
    }
}

MTLTextureSwizzle SwizzleOf( VkComponentSwizzle s, MTLTextureSwizzle identity )
{
    switch( s )
    {
        case VK_COMPONENT_SWIZZLE_ZERO: return MTLTextureSwizzleZero;
        case VK_COMPONENT_SWIZZLE_ONE: return MTLTextureSwizzleOne;
        case VK_COMPONENT_SWIZZLE_R: return MTLTextureSwizzleRed;
        case VK_COMPONENT_SWIZZLE_G: return MTLTextureSwizzleGreen;
        case VK_COMPONENT_SWIZZLE_B: return MTLTextureSwizzleBlue;
        case VK_COMPONENT_SWIZZLE_A: return MTLTextureSwizzleAlpha;
        default: return identity;
    }
}

id< MTLBuffer > EnsureMemoryBuffer( NativeMemory* m );
}

namespace
{
id< MTLBuffer > EnsureMemoryBuffer( NativeMemory* m )
{
    if( !m->buffer )
    {
        m->buffer = [Device() newBufferWithLength:std::max< VkDeviceSize >( m->size, 16 )
                                          options:m->hostVisible ? MTLResourceStorageModeShared
                                                                 : MTLResourceStorageModePrivate];
        m->buffer.label = m->hostVisible ? @"RTGL1 host visible memory" : @"RTGL1 device memory";
    }
    return m->buffer;
}

}

// ================================================================================================

bool RTGL1::MetalNative::Enabled()
{
    static const bool enabled = [] {
        const char* e = getenv( "RTGL1_METAL_ONLY" );
        return e && e[ 0 ] == '1';
    }();
    return enabled;
}

id< MTLDevice > RTGL1::MetalNative::Device()
{
    if( !G().device )
    {
        G().device = MTLCreateSystemDefaultDevice();
    }
    return G().device;
}

id< MTLCommandQueue > RTGL1::MetalNative::Queue()
{
    if( !G().queue )
    {
        G().queue       = [Device() newCommandQueueWithMaxCommandBufferCount:256];
        G().queue.label = @"RTGL1 (Metal only)";
    }
    return G().queue;
}

void RTGL1::MetalNative::NoteCommitted( id< MTLCommandBuffer > cb )
{
    std::lock_guard g( G().pendingLock );
    auto&           p = G().pending;
    p.erase( std::remove_if( p.begin(),
                             p.end(),
                             []( id< MTLCommandBuffer > c ) {
                                 return c.status == MTLCommandBufferStatusCompleted ||
                                        c.status == MTLCommandBufferStatusError;
                             } ),
             p.end() );
    p.push_back( cb );
}

VkDeviceMemory RTGL1::MetalNative::MemoryOf( VkBuffer buffer, VkDeviceSize& offset )
{
    auto* b = Obj< NativeBuffer >( buffer );
    if( !b || !b->memory )
    {
        return VK_NULL_HANDLE;
    }
    offset = b->offset;
    return Handle< VkDeviceMemory >( b->memory );
}

id< MTLTexture > RTGL1::MetalNative::TextureOf( VkImage image )
{
    auto* img = Obj< NativeImage >( image );
    return img && !img->swapchain ? img->texture : nil;
}

void RTGL1::MetalNative::WaitIdle()
{
    std::vector< id< MTLCommandBuffer > > p;
    {
        std::lock_guard g( G().pendingLock );
        p.swap( G().pending );
    }
    for( id< MTLCommandBuffer > cb : p )
    {
        [cb waitUntilCompleted];
    }
}

// ---- formats

RTGL1::MetalNative::FormatInfo RTGL1::MetalNative::GetFormatInfo( VkFormat f )
{
    FormatInfo i;
    auto       px = [ & ]( MTLPixelFormat m, uint32_t bytes, bool srgb = false ) {
        i.mtl           = m;
        i.bytesPerBlock = bytes;
        i.srgb          = srgb;
    };
    auto bc = [ & ]( MTLPixelFormat m, uint32_t bytes, bool srgb = false ) {
        px( m, bytes, srgb );
        i.blockWidth  = 4;
        i.blockHeight = 4;
        i.compressed  = true;
    };
    switch( f )
    {
        case VK_FORMAT_R8_UNORM: px( MTLPixelFormatR8Unorm, 1 ); break;
        case VK_FORMAT_R8_SNORM: px( MTLPixelFormatR8Snorm, 1 ); break;
        case VK_FORMAT_R8_UINT: px( MTLPixelFormatR8Uint, 1 ); break;
        case VK_FORMAT_R8_SINT: px( MTLPixelFormatR8Sint, 1 ); break;
        case VK_FORMAT_R8_SRGB: px( MTLPixelFormatR8Unorm_sRGB, 1, true ); break;
        case VK_FORMAT_R8G8_UNORM: px( MTLPixelFormatRG8Unorm, 2 ); break;
        case VK_FORMAT_R8G8_SNORM: px( MTLPixelFormatRG8Snorm, 2 ); break;
        case VK_FORMAT_R8G8_UINT: px( MTLPixelFormatRG8Uint, 2 ); break;
        case VK_FORMAT_R8G8_SINT: px( MTLPixelFormatRG8Sint, 2 ); break;
        case VK_FORMAT_R8G8_SRGB: px( MTLPixelFormatRG8Unorm_sRGB, 2, true ); break;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32: px( MTLPixelFormatRGBA8Unorm, 4 ); break;
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32: px( MTLPixelFormatRGBA8Snorm, 4 ); break;
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32: px( MTLPixelFormatRGBA8Uint, 4 ); break;
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_A8B8G8R8_SINT_PACK32: px( MTLPixelFormatRGBA8Sint, 4 ); break;
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32: px( MTLPixelFormatRGBA8Unorm_sRGB, 4, true ); break;
        case VK_FORMAT_B8G8R8A8_UNORM: px( MTLPixelFormatBGRA8Unorm, 4 ); break;
        case VK_FORMAT_B8G8R8A8_SRGB: px( MTLPixelFormatBGRA8Unorm_sRGB, 4, true ); break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: px( MTLPixelFormatRGB10A2Unorm, 4 ); break;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: px( MTLPixelFormatRGB10A2Uint, 4 ); break;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: px( MTLPixelFormatRG11B10Float, 4 ); break;
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: px( MTLPixelFormatRGB9E5Float, 4 ); break;
        case VK_FORMAT_R16_UNORM: px( MTLPixelFormatR16Unorm, 2 ); break;
        case VK_FORMAT_R16_SNORM: px( MTLPixelFormatR16Snorm, 2 ); break;
        case VK_FORMAT_R16_UINT: px( MTLPixelFormatR16Uint, 2 ); break;
        case VK_FORMAT_R16_SINT: px( MTLPixelFormatR16Sint, 2 ); break;
        case VK_FORMAT_R16_SFLOAT: px( MTLPixelFormatR16Float, 2 ); break;
        case VK_FORMAT_R16G16_UNORM: px( MTLPixelFormatRG16Unorm, 4 ); break;
        case VK_FORMAT_R16G16_SNORM: px( MTLPixelFormatRG16Snorm, 4 ); break;
        case VK_FORMAT_R16G16_UINT: px( MTLPixelFormatRG16Uint, 4 ); break;
        case VK_FORMAT_R16G16_SINT: px( MTLPixelFormatRG16Sint, 4 ); break;
        case VK_FORMAT_R16G16_SFLOAT: px( MTLPixelFormatRG16Float, 4 ); break;
        case VK_FORMAT_R16G16B16A16_UNORM: px( MTLPixelFormatRGBA16Unorm, 8 ); break;
        case VK_FORMAT_R16G16B16A16_SNORM: px( MTLPixelFormatRGBA16Snorm, 8 ); break;
        case VK_FORMAT_R16G16B16A16_UINT: px( MTLPixelFormatRGBA16Uint, 8 ); break;
        case VK_FORMAT_R16G16B16A16_SINT: px( MTLPixelFormatRGBA16Sint, 8 ); break;
        case VK_FORMAT_R16G16B16A16_SFLOAT: px( MTLPixelFormatRGBA16Float, 8 ); break;
        case VK_FORMAT_R32_UINT: px( MTLPixelFormatR32Uint, 4 ); break;
        case VK_FORMAT_R32_SINT: px( MTLPixelFormatR32Sint, 4 ); break;
        case VK_FORMAT_R32_SFLOAT: px( MTLPixelFormatR32Float, 4 ); break;
        case VK_FORMAT_R32G32_UINT: px( MTLPixelFormatRG32Uint, 8 ); break;
        case VK_FORMAT_R32G32_SINT: px( MTLPixelFormatRG32Sint, 8 ); break;
        case VK_FORMAT_R32G32_SFLOAT: px( MTLPixelFormatRG32Float, 8 ); break;
        case VK_FORMAT_R32G32B32A32_UINT: px( MTLPixelFormatRGBA32Uint, 16 ); break;
        case VK_FORMAT_R32G32B32A32_SINT: px( MTLPixelFormatRGBA32Sint, 16 ); break;
        case VK_FORMAT_R32G32B32A32_SFLOAT: px( MTLPixelFormatRGBA32Float, 16 ); break;

        case VK_FORMAT_D16_UNORM:
            px( MTLPixelFormatDepth16Unorm, 2 );
            i.depth = true;
            break;
        case VK_FORMAT_D32_SFLOAT:
            px( MTLPixelFormatDepth32Float, 4 );
            i.depth = true;
            break;
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            px( MTLPixelFormatDepth32Float_Stencil8, 8 );
            i.depth   = true;
            i.stencil = true;
            break;
        case VK_FORMAT_S8_UINT:
            px( MTLPixelFormatStencil8, 1 );
            i.stencil = true;
            break;

        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: bc( MTLPixelFormatBC1_RGBA, 8 ); break;
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: bc( MTLPixelFormatBC1_RGBA_sRGB, 8, true ); break;
        case VK_FORMAT_BC2_UNORM_BLOCK: bc( MTLPixelFormatBC2_RGBA, 16 ); break;
        case VK_FORMAT_BC2_SRGB_BLOCK: bc( MTLPixelFormatBC2_RGBA_sRGB, 16, true ); break;
        case VK_FORMAT_BC3_UNORM_BLOCK: bc( MTLPixelFormatBC3_RGBA, 16 ); break;
        case VK_FORMAT_BC3_SRGB_BLOCK: bc( MTLPixelFormatBC3_RGBA_sRGB, 16, true ); break;
        case VK_FORMAT_BC4_UNORM_BLOCK: bc( MTLPixelFormatBC4_RUnorm, 8 ); break;
        case VK_FORMAT_BC4_SNORM_BLOCK: bc( MTLPixelFormatBC4_RSnorm, 8 ); break;
        case VK_FORMAT_BC5_UNORM_BLOCK: bc( MTLPixelFormatBC5_RGUnorm, 16 ); break;
        case VK_FORMAT_BC5_SNORM_BLOCK: bc( MTLPixelFormatBC5_RGSnorm, 16 ); break;
        case VK_FORMAT_BC6H_UFLOAT_BLOCK: bc( MTLPixelFormatBC6H_RGBUfloat, 16 ); break;
        case VK_FORMAT_BC6H_SFLOAT_BLOCK: bc( MTLPixelFormatBC6H_RGBFloat, 16 ); break;
        case VK_FORMAT_BC7_UNORM_BLOCK: bc( MTLPixelFormatBC7_RGBAUnorm, 16 ); break;
        case VK_FORMAT_BC7_SRGB_BLOCK: bc( MTLPixelFormatBC7_RGBAUnorm_sRGB, 16, true ); break;
        default: break;
    }
    return i;
}

// ================================================================================================
// instance, physical device, device

VkResult RTGL1::MetalNative::vkCreateInstance( const VkInstanceCreateInfo*  ci,
                                               const VkAllocationCallbacks* alloc,
                                               VkInstance*                  out )
{
    if( !Device() || !Device().supportsRaytracing )
    {
        Log( "no Metal device with ray tracing" );
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
#ifndef RG_HAS_SPIRV_CROSS
    // every shader is translated to Metal, there's no other way to run them here
    Log( "built without SPIRV-Cross (RG_SPIRV_CROSS_DIR), Metal only mode is unavailable" );
    return VK_ERROR_INCOMPATIBLE_DRIVER;
#endif
    if( @available( macOS 15.0, * ) )
    {
    }
    else
    {
        Log( "macOS 15 or newer is needed" );
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    char buf[ 256 ];
    snprintf( buf, sizeof( buf ), "Vulkan implemented on Metal (no MoltenVK), device: %s", Device().name.UTF8String );
    Log( buf );
    *out = reinterpret_cast< VkInstance >( &g_instance );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyInstance( VkInstance, const VkAllocationCallbacks* ) {}

VkResult RTGL1::MetalNative::vkEnumerateInstanceExtensionProperties( const char*            layer,
                                                                     uint32_t*              count,
                                                                     VkExtensionProperties* out )
{
    const VkExtensionProperties exts[] = {
        Ext( VK_KHR_SURFACE_EXTENSION_NAME ),
        Ext( VK_EXT_METAL_SURFACE_EXTENSION_NAME ),
        Ext( VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME ),
        Ext( VK_EXT_DEBUG_UTILS_EXTENSION_NAME ),
        Ext( "VK_KHR_portability_enumeration" ),
    };
    if( layer )
    {
        *count = 0;
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    return FillArray( count, out, exts, uint32_t( std::size( exts ) ) );
}

VkResult RTGL1::MetalNative::vkEnumerateInstanceLayerProperties( uint32_t* count, VkLayerProperties* )
{
    *count = 0;
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkEnumeratePhysicalDevices( VkInstance, uint32_t* count, VkPhysicalDevice* out )
{
    const VkPhysicalDevice p = reinterpret_cast< VkPhysicalDevice >( &g_physicalDevice );
    return FillArray( count, out, &p, 1 );
}

void RTGL1::MetalNative::vkGetPhysicalDeviceProperties( VkPhysicalDevice, VkPhysicalDeviceProperties* p )
{
    *p               = {};
    p->apiVersion    = VK_API_VERSION_1_2;
    p->driverVersion = 1;
    p->vendorID      = 0x106B;
    p->deviceID      = 1;
    p->deviceType    = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    strncpy( p->deviceName, Device().name.UTF8String, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1 );

    VkPhysicalDeviceLimits& l                          = p->limits;
    l.maxImageDimension1D                              = 16384;
    l.maxImageDimension2D                              = 16384;
    l.maxImageDimension3D                              = 2048;
    l.maxImageDimensionCube                            = 16384;
    l.maxImageArrayLayers                              = 2048;
    l.maxTexelBufferElements                           = 1u << 26;
    l.maxUniformBufferRange                            = 1u << 30;
    l.maxStorageBufferRange                            = 1u << 30;
    l.maxPushConstantsSize                             = 4096;
    l.maxMemoryAllocationCount                         = 1u << 20;
    l.maxSamplerAllocationCount                        = 1u << 16;
    l.bufferImageGranularity                           = 1;
    l.maxBoundDescriptorSets                           = 16;
    l.maxPerStageDescriptorSamplers                    = 1u << 20;
    l.maxPerStageDescriptorUniformBuffers              = 1u << 20;
    l.maxPerStageDescriptorStorageBuffers              = 1u << 20;
    l.maxPerStageDescriptorSampledImages               = 1u << 20;
    l.maxPerStageDescriptorStorageImages               = 1u << 20;
    l.maxPerStageDescriptorInputAttachments            = 8;
    l.maxPerStageResources                             = 1u << 20;
    l.maxDescriptorSetSamplers                         = 1u << 20;
    l.maxDescriptorSetUniformBuffers                   = 1u << 20;
    l.maxDescriptorSetUniformBuffersDynamic            = 64;
    l.maxDescriptorSetStorageBuffers                   = 1u << 20;
    l.maxDescriptorSetStorageBuffersDynamic            = 64;
    l.maxDescriptorSetSampledImages                    = 1u << 20;
    l.maxDescriptorSetStorageImages                    = 1u << 20;
    l.maxDescriptorSetInputAttachments                 = 8;
    l.maxVertexInputAttributes                         = 31;
    l.maxVertexInputBindings                           = 8;
    l.maxVertexInputAttributeOffset                    = 2047;
    l.maxVertexInputBindingStride                      = 2048;
    l.maxVertexOutputComponents                        = 124;
    l.maxFragmentInputComponents                       = 124;
    l.maxFragmentOutputAttachments                     = 8;
    l.maxFragmentCombinedOutputResources               = 1u << 20;
    l.maxComputeSharedMemorySize                       = 32768;
    l.maxComputeWorkGroupCount[ 0 ]                    = 65535;
    l.maxComputeWorkGroupCount[ 1 ]                    = 65535;
    l.maxComputeWorkGroupCount[ 2 ]                    = 65535;
    l.maxComputeWorkGroupInvocations                   = 1024;
    l.maxComputeWorkGroupSize[ 0 ]                     = 1024;
    l.maxComputeWorkGroupSize[ 1 ]                     = 1024;
    l.maxComputeWorkGroupSize[ 2 ]                     = 1024;
    l.subPixelPrecisionBits                            = 4;
    l.subTexelPrecisionBits                            = 4;
    l.mipmapPrecisionBits                              = 4;
    l.maxDrawIndexedIndexValue                         = UINT32_MAX;
    l.maxDrawIndirectCount                             = UINT32_MAX;
    l.maxSamplerLodBias                                = 4;
    l.maxSamplerAnisotropy                             = 16;
    l.maxViewports                                     = 16;
    l.maxViewportDimensions[ 0 ]                       = 16384;
    l.maxViewportDimensions[ 1 ]                       = 16384;
    l.viewportBoundsRange[ 0 ]                         = -32768;
    l.viewportBoundsRange[ 1 ]                         = 32767;
    l.minMemoryMapAlignment                            = 64;
    l.minTexelBufferOffsetAlignment                    = 16;
    l.minUniformBufferOffsetAlignment                  = 256;
    l.minStorageBufferOffsetAlignment                  = 16;
    l.maxTexelOffset                                   = 7;
    l.minTexelOffset                                   = -8;
    l.maxTexelGatherOffset                             = 7;
    l.minTexelGatherOffset                             = -8;
    l.maxFramebufferWidth                              = 16384;
    l.maxFramebufferHeight                             = 16384;
    l.maxFramebufferLayers                             = 2048;
    l.framebufferColorSampleCounts                     = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    l.framebufferDepthSampleCounts                     = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    l.framebufferStencilSampleCounts                   = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    l.framebufferNoAttachmentsSampleCounts             = VK_SAMPLE_COUNT_1_BIT;
    l.maxColorAttachments                              = 8;
    l.sampledImageColorSampleCounts                    = VK_SAMPLE_COUNT_1_BIT;
    l.sampledImageIntegerSampleCounts                  = VK_SAMPLE_COUNT_1_BIT;
    l.sampledImageDepthSampleCounts                    = VK_SAMPLE_COUNT_1_BIT;
    l.sampledImageStencilSampleCounts                  = VK_SAMPLE_COUNT_1_BIT;
    l.storageImageSampleCounts                         = VK_SAMPLE_COUNT_1_BIT;
    l.maxSampleMaskWords                               = 1;
    l.timestampComputeAndGraphics                      = VK_TRUE;
    l.timestampPeriod                                  = 1.0f;
    l.maxClipDistances                                 = 8;
    l.maxCullDistances                                 = 8;
    l.maxCombinedClipAndCullDistances                  = 8;
    l.discreteQueuePriorities                          = 2;
    l.pointSizeRange[ 0 ]                              = 1;
    l.pointSizeRange[ 1 ]                              = 64;
    l.lineWidthRange[ 0 ]                              = 1;
    l.lineWidthRange[ 1 ]                              = 1;
    l.pointSizeGranularity                             = 1;
    l.lineWidthGranularity                             = 1;
    l.standardSampleLocations                          = VK_TRUE;
    l.optimalBufferCopyOffsetAlignment                 = 16;
    l.optimalBufferCopyRowPitchAlignment               = 1;
    l.nonCoherentAtomSize                              = 64;
}

void RTGL1::MetalNative::vkGetPhysicalDeviceProperties2( VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p )
{
    RTGL1::MetalNative::vkGetPhysicalDeviceProperties( pd, &p->properties );
    for( auto* n = static_cast< VkBaseOutStructure* >( p->pNext ); n; n = n->pNext )
    {
        switch( n->sType )
        {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR: {
                auto* r                               = reinterpret_cast< VkPhysicalDeviceRayTracingPipelinePropertiesKHR* >( n );
                r->shaderGroupHandleSize              = 32;
                r->maxRayRecursionDepth               = 31;
                r->maxShaderGroupStride               = 4096;
                r->shaderGroupBaseAlignment           = 64;
                r->shaderGroupHandleCaptureReplaySize = 32;
                r->maxRayDispatchInvocationCount      = 1u << 30;
                r->shaderGroupHandleAlignment         = 32;
                r->maxRayHitAttributeSize             = 32;
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR: {
                auto* a                       = reinterpret_cast< VkPhysicalDeviceAccelerationStructurePropertiesKHR* >( n );
                a->maxGeometryCount           = 1u << 24;
                a->maxInstanceCount           = 1u << 24;
                a->maxPrimitiveCount          = 1u << 29;
                a->maxPerStageDescriptorAccelerationStructures                = 16;
                a->maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 16;
                a->maxDescriptorSetAccelerationStructures                     = 16;
                a->maxDescriptorSetUpdateAfterBindAccelerationStructures      = 16;
                a->minAccelerationStructureScratchOffsetAlignment             = 256;
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: {
                auto* s                      = reinterpret_cast< VkPhysicalDeviceSubgroupProperties* >( n );
                s->subgroupSize              = 32;
                s->supportedStages           = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;
                s->supportedOperations       = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                         VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                         VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
                                         VK_SUBGROUP_FEATURE_QUAD_BIT;
                s->quadOperationsInAllStages = VK_TRUE;
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: {
                auto* d     = reinterpret_cast< VkPhysicalDeviceDriverProperties* >( n );
                d->driverID = VK_DRIVER_ID_MOLTENVK;
                strncpy( d->driverName, "RTGL1 Metal", VK_MAX_DRIVER_NAME_SIZE - 1 );
                strncpy( d->driverInfo, "Vulkan subset on Metal", VK_MAX_DRIVER_INFO_SIZE - 1 );
                break;
            }
            default: break;
        }
    }
}

void RTGL1::MetalNative::vkGetPhysicalDeviceFeatures( VkPhysicalDevice, VkPhysicalDeviceFeatures* f )
{
    *f = {};
    EnableAllFeatures( reinterpret_cast< uint8_t* >( f ) - offsetof( VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge ),
                       offsetof( VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge ) + sizeof( *f ) );
    f->geometryShader     = VK_FALSE;
    f->tessellationShader = VK_FALSE;
    f->sparseBinding      = VK_FALSE;
}

void RTGL1::MetalNative::vkGetPhysicalDeviceFeatures2( VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* f )
{
    RTGL1::MetalNative::vkGetPhysicalDeviceFeatures( pd, &f->features );
    for( auto* n = static_cast< VkBaseOutStructure* >( f->pNext ); n; n = n->pNext )
    {
        size_t size = 0;
        switch( n->sType )
        {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: size = sizeof( VkPhysicalDeviceVulkan11Features ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: size = sizeof( VkPhysicalDeviceVulkan12Features ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES: size = sizeof( VkPhysicalDeviceMultiviewFeatures ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: size = sizeof( VkPhysicalDevice16BitStorageFeatures ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: size = sizeof( VkPhysicalDevice8BitStorageFeatures ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR: size = sizeof( VkPhysicalDeviceSynchronization2FeaturesKHR ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR: size = sizeof( VkPhysicalDeviceRayTracingPipelineFeaturesKHR ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR: size = sizeof( VkPhysicalDeviceRayQueryFeaturesKHR ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR: size = sizeof( VkPhysicalDeviceAccelerationStructureFeaturesKHR ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: size = sizeof( VkPhysicalDeviceBufferDeviceAddressFeatures ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: size = sizeof( VkPhysicalDeviceTimelineSemaphoreFeatures ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES: size = sizeof( VkPhysicalDeviceDescriptorIndexingFeatures ); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: size = sizeof( VkPhysicalDeviceShaderFloat16Int8Features ); break;
            default: break;
        }
        if( size > 0 )
        {
            EnableAllFeatures( n, size );
        }
    }
}

void RTGL1::MetalNative::vkGetPhysicalDeviceMemoryProperties( VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* m )
{
    *m                            = {};
    m->memoryHeapCount            = 1;
    m->memoryHeaps[ 0 ].size      = std::max< uint64_t >( Device().recommendedMaxWorkingSetSize, 1ull << 30 );
    m->memoryHeaps[ 0 ].flags     = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    m->memoryTypeCount            = 2;
    m->memoryTypes[ 0 ].heapIndex = 0;
    m->memoryTypes[ 0 ].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    m->memoryTypes[ 1 ].heapIndex     = 0;
    m->memoryTypes[ 1 ].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
}

void RTGL1::MetalNative::vkGetPhysicalDeviceMemoryProperties2( VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* m )
{
    RTGL1::MetalNative::vkGetPhysicalDeviceMemoryProperties( pd, &m->memoryProperties );
}

void RTGL1::MetalNative::vkGetPhysicalDeviceQueueFamilyProperties( VkPhysicalDevice,
                                                                   uint32_t*                count,
                                                                   VkQueueFamilyProperties* out )
{
    VkQueueFamilyProperties q        = {};
    q.queueFlags                     = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    q.queueCount                     = 1;
    q.timestampValidBits             = 64;
    q.minImageTransferGranularity    = { 1, 1, 1 };
    FillArray( count, out, &q, 1 );
}

void RTGL1::MetalNative::vkGetPhysicalDeviceFormatProperties( VkPhysicalDevice, VkFormat format, VkFormatProperties* p )
{
    *p                  = {};
    const FormatInfo fi = GetFormatInfo( format );
    if( fi.mtl != MTLPixelFormatInvalid )
    {
        VkFormatFeatureFlags f = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                 VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                 VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        if( fi.depth || fi.stencil )
        {
            f |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }
        else if( !fi.compressed )
        {
            f |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                 VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        }
        p->optimalTilingFeatures = f;
    }
    switch( format )
    {
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32B32_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            p->bufferFeatures = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT |
                                VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR;
            break;
        default: break;
    }
}

VkResult RTGL1::MetalNative::vkEnumerateDeviceExtensionProperties( VkPhysicalDevice,
                                                                   const char*            layer,
                                                                   uint32_t*              count,
                                                                   VkExtensionProperties* out )
{
    const VkExtensionProperties exts[] = {
        Ext( VK_KHR_SWAPCHAIN_EXTENSION_NAME ),
        Ext( VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME ),
        Ext( VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME ),
        Ext( VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME ),
        Ext( VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME ),
        Ext( VK_KHR_RAY_QUERY_EXTENSION_NAME ),
        Ext( VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME ),
        Ext( VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME ),
        Ext( VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME ),
        Ext( VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME ),
        Ext( VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME ),
        Ext( VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME ),
        Ext( VK_KHR_BIND_MEMORY_2_EXTENSION_NAME ),
        Ext( VK_KHR_MULTIVIEW_EXTENSION_NAME ),
        Ext( VK_EXT_METAL_OBJECTS_EXTENSION_NAME ),
    };
    if( layer )
    {
        *count = 0;
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    return FillArray( count, out, exts, uint32_t( std::size( exts ) ) );
}

VkResult RTGL1::MetalNative::vkCreateDevice( VkPhysicalDevice, const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice* out )
{
    Queue();
    *out = reinterpret_cast< VkDevice >( &g_device );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyDevice( VkDevice, const VkAllocationCallbacks* )
{
    WaitIdle();
}

void RTGL1::MetalNative::vkGetDeviceQueue( VkDevice, uint32_t, uint32_t, VkQueue* out )
{
    *out = reinterpret_cast< VkQueue >( &g_queue );
}

namespace
{
PFN_vkVoidFunction NativeProc( const char* name )
{
    if( !name )
    {
        return nullptr;
    }
    static const std::unordered_map< std::string, PFN_vkVoidFunction > table = [] {
        std::unordered_map< std::string, PFN_vkVoidFunction > t;
#define RG_NATIVE_PROC( n ) t[ #n ] = reinterpret_cast< PFN_vkVoidFunction >( &RTGL1::MetalNative::n );
        RG_NATIVE_FUNCTIONS( RG_NATIVE_PROC )
#undef RG_NATIVE_PROC
        // promoted to core; the extension names too
        t[ "vkGetBufferMemoryRequirements2KHR" ]     = t[ "vkGetBufferMemoryRequirements2" ];
        t[ "vkGetImageMemoryRequirements2KHR" ]      = t[ "vkGetImageMemoryRequirements2" ];
        t[ "vkBindBufferMemory2KHR" ]                = t[ "vkBindBufferMemory2" ];
        t[ "vkBindImageMemory2KHR" ]                 = t[ "vkBindImageMemory2" ];
        t[ "vkGetPhysicalDeviceMemoryProperties2KHR" ] = t[ "vkGetPhysicalDeviceMemoryProperties2" ];
        t[ "vkGetPhysicalDeviceProperties2KHR" ]     = t[ "vkGetPhysicalDeviceProperties2" ];
        t[ "vkGetPhysicalDeviceFeatures2KHR" ]       = t[ "vkGetPhysicalDeviceFeatures2" ];
        t[ "vkGetBufferDeviceAddressKHR" ]           = t[ "vkGetBufferDeviceAddress" ];
        t[ "vkCmdDrawIndexedIndirectCountKHR" ]      = t[ "vkCmdDrawIndexedIndirectCount" ];
        t[ "vkCmdPipelineBarrier2" ]                 = t[ "vkCmdPipelineBarrier2KHR" ];
        return t;
    }();
    auto it = table.find( name );
    return it != table.end() ? it->second : nullptr;
}
}

PFN_vkVoidFunction RTGL1::MetalNative::vkGetDeviceProcAddr( VkDevice, const char* name )
{
    return NativeProc( name );
}

PFN_vkVoidFunction RTGL1::MetalNative::vkGetInstanceProcAddr( VkInstance, const char* name )
{
    return NativeProc( name );
}

VkResult RTGL1::MetalNative::vkDeviceWaitIdle( VkDevice )
{
    WaitIdle();
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkQueueWaitIdle( VkQueue )
{
    WaitIdle();
    return VK_SUCCESS;
}

// ================================================================================================
// surface, swapchain

VkResult RTGL1::MetalNative::vkCreateMetalSurfaceEXT( VkInstance,
                                                      const VkMetalSurfaceCreateInfoEXT* ci,
                                                      const VkAllocationCallbacks*,
                                                      VkSurfaceKHR* out )
{
    if( !ci->pLayer )
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto* s  = new NativeSurface;
    s->layer = ( CAMetalLayer* )ci->pLayer;
    *out     = Handle< VkSurfaceKHR >( s );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroySurfaceKHR( VkInstance, VkSurfaceKHR surface, const VkAllocationCallbacks* )
{
    delete Obj< NativeSurface >( surface );
}

VkResult RTGL1::MetalNative::vkGetPhysicalDeviceSurfaceSupportKHR( VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32* supported )
{
    *supported = VK_TRUE;
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkGetPhysicalDeviceSurfaceCapabilitiesKHR( VkPhysicalDevice,
                                                                        VkSurfaceKHR              surface,
                                                                        VkSurfaceCapabilitiesKHR* c )
{
    CAMetalLayer* layer = Obj< NativeSurface >( surface )->layer;
    const CGSize  size  = layer.bounds.size;
    const CGFloat scale = layer.contentsScale;

    *c                         = {};
    c->minImageCount           = 2;
    c->maxImageCount           = 3;
    c->currentExtent           = { uint32_t( std::max( 0.0, std::round( size.width * scale ) ) ),
                                   uint32_t( std::max( 0.0, std::round( size.height * scale ) ) ) };
    c->minImageExtent          = { 1, 1 };
    c->maxImageExtent          = { 16384, 16384 };
    c->maxImageArrayLayers     = 1;
    c->supportedTransforms     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    c->currentTransform        = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    c->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    c->supportedUsageFlags     = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if( c->currentExtent.width == 0 || c->currentExtent.height == 0 )
    {
        c->maxImageExtent = { 0, 0 };
    }
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkGetPhysicalDeviceSurfaceFormatsKHR( VkPhysicalDevice,
                                                                   VkSurfaceKHR,
                                                                   uint32_t*           count,
                                                                   VkSurfaceFormatKHR* out )
{
    const VkSurfaceFormatKHR formats[] = {
        { VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    };
    return FillArray( count, out, formats, uint32_t( std::size( formats ) ) );
}

VkResult RTGL1::MetalNative::vkGetPhysicalDeviceSurfacePresentModesKHR( VkPhysicalDevice,
                                                                        VkSurfaceKHR,
                                                                        uint32_t*         count,
                                                                        VkPresentModeKHR* out )
{
    const VkPresentModeKHR modes[] = { VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR };
    return FillArray( count, out, modes, uint32_t( std::size( modes ) ) );
}

VkResult RTGL1::MetalNative::vkCreateSwapchainKHR( VkDevice,
                                                   const VkSwapchainCreateInfoKHR* ci,
                                                   const VkAllocationCallbacks*,
                                                   VkSwapchainKHR* out )
{
    auto* sc    = new NativeSwapchain;
    sc->surface = Obj< NativeSurface >( ci->surface );
    sc->format  = ci->imageFormat;
    sc->extent  = ci->imageExtent;

    CAMetalLayer* layer = sc->surface->layer;
    layer.device          = Device();
    // the sRGB encoding is done by the shaders that write the swapchain images
    layer.pixelFormat     = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;
    layer.drawableSize    = CGSizeMake( ci->imageExtent.width, ci->imageExtent.height );
    layer.displaySyncEnabled   = ci->presentMode != VK_PRESENT_MODE_IMMEDIATE_KHR;
    layer.maximumDrawableCount = 3;
    {
        CGColorSpaceRef cs = CGColorSpaceCreateWithName( kCGColorSpaceSRGB );
        layer.colorspace   = cs;
        CGColorSpaceRelease( cs );
    }

    const uint32_t count = 3;
    for( uint32_t i = 0; i < count; i++ )
    {
        auto* img           = new NativeImage;
        img->format         = ci->imageFormat;
        img->extent         = { ci->imageExtent.width, ci->imageExtent.height, 1 };
        img->usage          = ci->imageUsage;
        img->swapchain      = sc;
        img->swapchainIndex = i;
        sc->images.push_back( img );
    }
    sc->drawables.resize( count );
    *out = Handle< VkSwapchainKHR >( sc );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroySwapchainKHR( VkDevice, VkSwapchainKHR swapchain, const VkAllocationCallbacks* )
{
    auto* sc = Obj< NativeSwapchain >( swapchain );
    if( !sc )
    {
        return;
    }
    for( NativeImage* img : sc->images )
    {
        delete img;
    }
    delete sc;
}

VkResult RTGL1::MetalNative::vkGetSwapchainImagesKHR( VkDevice, VkSwapchainKHR swapchain, uint32_t* count, VkImage* out )
{
    auto*                  sc = Obj< NativeSwapchain >( swapchain );
    std::vector< VkImage > images;
    for( NativeImage* img : sc->images )
    {
        images.push_back( Handle< VkImage >( img ) );
    }
    return FillArray( count, out, images.data(), uint32_t( images.size() ) );
}

VkResult RTGL1::MetalNative::vkAcquireNextImageKHR( VkDevice,
                                                    VkSwapchainKHR swapchain,
                                                    uint64_t,
                                                    VkSemaphore semaphore,
                                                    VkFence     fence,
                                                    uint32_t*   index )
{
    @autoreleasepool
    {
        auto*                sc       = Obj< NativeSwapchain >( swapchain );
        id< CAMetalDrawable > drawable = [sc->surface->layer nextDrawable];
        if( !drawable )
        {
            // e.g. the window is being resized: try again
            return VK_ERROR_OUT_OF_DATE_KHR;
        }
        const uint32_t i   = sc->next;
        sc->next           = ( sc->next + 1 ) % uint32_t( sc->images.size() );
        sc->drawables[ i ] = drawable;
        *index             = i;

        if( auto* s = Obj< NativeSemaphore >( semaphore ) )
        {
            // available right away
            s->value++;
            s->event.signaledValue = s->value;
        }
        if( auto* f = Obj< NativeFence >( fence ) )
        {
            std::lock_guard g( f->lock );
            f->signaled = true;
            f->cv.notify_all();
        }
        return VK_SUCCESS;
    }
}

VkResult RTGL1::MetalNative::vkQueuePresentKHR( VkQueue, const VkPresentInfoKHR* pi )
{
    @autoreleasepool
    {
        id< MTLCommandBuffer > cb = [Queue() commandBuffer];
        cb.label                  = @"RTGL1 present";
        for( uint32_t i = 0; i < pi->waitSemaphoreCount; i++ )
        {
            if( auto* s = Obj< NativeSemaphore >( pi->pWaitSemaphores[ i ] ) )
            {
                [cb encodeWaitForEvent:s->event value:s->value];
            }
        }
        for( uint32_t i = 0; i < pi->swapchainCount; i++ )
        {
            auto*          sc  = Obj< NativeSwapchain >( pi->pSwapchains[ i ] );
            const uint32_t idx = pi->pImageIndices[ i ];
            if( idx < sc->drawables.size() && sc->drawables[ idx ] )
            {
                [cb presentDrawable:sc->drawables[ idx ]];
                sc->drawables[ idx ] = nil;
            }
            if( pi->pResults )
            {
                pi->pResults[ i ] = VK_SUCCESS;
            }
        }
        [cb commit];
        NoteCommitted( cb );
        return VK_SUCCESS;
    }
}

id< MTLTexture > RTGL1::MetalNative::SwapchainTexture( VkImage image )
{
    auto* img = Obj< NativeImage >( image );
    if( !Enabled() || !img || !img->swapchain )
    {
        return nil;
    }
    id< CAMetalDrawable > d = img->swapchain->drawables[ img->swapchainIndex ];
    return d ? d.texture : nil;
}

bool RTGL1::MetalNative::NeedsSrgbEncode( VkImage image )
{
    auto* img = Obj< NativeImage >( image );
    return Enabled() && img && img->swapchain && GetFormatInfo( img->format ).srgb;
}

VkFormat RTGL1::MetalNative::ImageFormat( VkImage image )
{
    auto* img = Obj< NativeImage >( image );
    return Enabled() && img ? img->format : VK_FORMAT_UNDEFINED;
}

// ================================================================================================
// memory, buffers, images

VkResult RTGL1::MetalNative::vkAllocateMemory( VkDevice,
                                               const VkMemoryAllocateInfo* ai,
                                               const VkAllocationCallbacks*,
                                               VkDeviceMemory* out )
{
    auto* m        = new NativeMemory;
    m->size        = ai->allocationSize;
    m->typeIndex   = ai->memoryTypeIndex;
    m->hostVisible = ai->memoryTypeIndex == 1;
    if( m->hostVisible )
    {
        if( !EnsureMemoryBuffer( m ) )
        {
            delete m;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    *out = Handle< VkDeviceMemory >( m );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkFreeMemory( VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks* )
{
    RTGL1::Metal::ForgetMemory( memory ); // also when freed by VMA, bypassing the shims
    delete Obj< NativeMemory >( memory );
}

VkResult RTGL1::MetalNative::vkMapMemory( VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize, VkMemoryMapFlags, void** out )
{
    auto* m = Obj< NativeMemory >( memory );
    if( !m->hostVisible )
    {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    *out = static_cast< uint8_t* >( EnsureMemoryBuffer( m ).contents ) + offset;
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkUnmapMemory( VkDevice, VkDeviceMemory ) {}

VkResult RTGL1::MetalNative::vkFlushMappedMemoryRanges( VkDevice, uint32_t, const VkMappedMemoryRange* )
{
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkInvalidateMappedMemoryRanges( VkDevice, uint32_t, const VkMappedMemoryRange* )
{
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkBindBufferMemory( VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset )
{
    auto* b   = Obj< NativeBuffer >( buffer );
    b->memory = Obj< NativeMemory >( memory );
    b->offset = offset;
    return EnsureMemoryBuffer( b->memory ) ? VK_SUCCESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult RTGL1::MetalNative::vkBindImageMemory( VkDevice, VkImage, VkDeviceMemory, VkDeviceSize )
{
    return VK_SUCCESS; // images have their own textures
}

VkResult RTGL1::MetalNative::vkBindBufferMemory2( VkDevice device, uint32_t count, const VkBindBufferMemoryInfo* infos )
{
    for( uint32_t i = 0; i < count; i++ )
    {
        const VkResult r = RTGL1::MetalNative::vkBindBufferMemory( device, infos[ i ].buffer, infos[ i ].memory, infos[ i ].memoryOffset );
        if( r != VK_SUCCESS )
        {
            return r;
        }
    }
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkBindImageMemory2( VkDevice, uint32_t, const VkBindImageMemoryInfo* )
{
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkGetBufferMemoryRequirements( VkDevice, VkBuffer buffer, VkMemoryRequirements* r )
{
    auto* b           = Obj< NativeBuffer >( buffer );
    r->size           = ( b->size + 255 ) & ~VkDeviceSize( 255 );
    r->alignment      = 256;
    r->memoryTypeBits = 0b11;
}

void RTGL1::MetalNative::vkGetImageMemoryRequirements( VkDevice, VkImage image, VkMemoryRequirements* r )
{
    auto* img         = Obj< NativeImage >( image );
    r->size           = std::max< NSUInteger >( img->memorySize, 256 );
    r->alignment      = std::max< NSUInteger >( img->memoryAlign, 256 );
    r->memoryTypeBits = 0b01;
}

namespace
{
void FillDedicated( void* pNext )
{
    for( auto* n = static_cast< VkBaseOutStructure* >( pNext ); n; n = n->pNext )
    {
        if( n->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS )
        {
            auto* d                        = reinterpret_cast< VkMemoryDedicatedRequirements* >( n );
            d->prefersDedicatedAllocation  = VK_FALSE;
            d->requiresDedicatedAllocation = VK_FALSE;
        }
    }
}
}

void RTGL1::MetalNative::vkGetBufferMemoryRequirements2( VkDevice                               device,
                                                         const VkBufferMemoryRequirementsInfo2* info,
                                                         VkMemoryRequirements2*                 r )
{
    RTGL1::MetalNative::vkGetBufferMemoryRequirements( device, info->buffer, &r->memoryRequirements );
    FillDedicated( r->pNext );
}

void RTGL1::MetalNative::vkGetImageMemoryRequirements2( VkDevice                              device,
                                                        const VkImageMemoryRequirementsInfo2* info,
                                                        VkMemoryRequirements2*                r )
{
    RTGL1::MetalNative::vkGetImageMemoryRequirements( device, info->image, &r->memoryRequirements );
    FillDedicated( r->pNext );
}

VkResult RTGL1::MetalNative::vkCreateBuffer( VkDevice, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*, VkBuffer* out )
{
    auto* b  = new NativeBuffer;
    b->size  = ci->size;
    b->usage = ci->usage;
    *out     = Handle< VkBuffer >( b );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyBuffer( VkDevice, VkBuffer buffer, const VkAllocationCallbacks* )
{
    delete Obj< NativeBuffer >( buffer );
}

VkDeviceAddress RTGL1::MetalNative::vkGetBufferDeviceAddress( VkDevice, const VkBufferDeviceAddressInfo* info )
{
    auto* b = Obj< NativeBuffer >( info->buffer );
    if( !b || !b->memory )
    {
        return 0;
    }
    return EnsureMemoryBuffer( b->memory ).gpuAddress + b->offset;
}

VkResult RTGL1::MetalNative::vkCreateImage( VkDevice, const VkImageCreateInfo* ci, const VkAllocationCallbacks*, VkImage* out )
{
    @autoreleasepool
    {
        const FormatInfo fi = GetFormatInfo( ci->format );
        if( fi.mtl == MTLPixelFormatInvalid )
        {
            char buf[ 96 ];
            snprintf( buf, sizeof( buf ), "image format %d is not supported", int( ci->format ) );
            Log( buf );
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }

        MTLTextureDescriptor* d = [MTLTextureDescriptor new];
        d.pixelFormat           = fi.mtl;
        d.width                 = ci->extent.width;
        d.height                = std::max( 1u, ci->extent.height );
        d.depth                 = std::max( 1u, ci->extent.depth );
        d.mipmapLevelCount      = std::max( 1u, ci->mipLevels );
        d.sampleCount           = ci->samples > 1 ? ci->samples : 1;
        d.storageMode           = MTLStorageModePrivate;

        const uint32_t layers = std::max( 1u, ci->arrayLayers );
        if( ci->imageType == VK_IMAGE_TYPE_3D )
        {
            d.textureType = MTLTextureType3D;
        }
        else if( ci->imageType == VK_IMAGE_TYPE_1D )
        {
            d.textureType = layers > 1 ? MTLTextureType1DArray : MTLTextureType1D;
            d.arrayLength = layers;
        }
        else if( ( ci->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT ) && layers % 6 == 0 )
        {
            d.textureType = layers == 6 ? MTLTextureTypeCube : MTLTextureTypeCubeArray;
            d.arrayLength = layers / 6;
        }
        else
        {
            d.textureType = layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
            d.arrayLength = layers;
        }

        MTLTextureUsage usage = MTLTextureUsageShaderRead;
        if( ci->usage & VK_IMAGE_USAGE_STORAGE_BIT )
        {
            usage |= MTLTextureUsageShaderWrite;
        }
        if( ci->usage & ( VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT ) )
        {
            usage |= MTLTextureUsageRenderTarget;
        }
        if( ( ci->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT ) && !fi.compressed && !fi.depth && !fi.stencil )
        {
            // blits (e.g. mipmap generation) are compute passes
            usage |= MTLTextureUsageShaderWrite;
        }
        if( ( ci->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT ) || ( fi.srgb && !fi.compressed ) )
        {
            // sRGB: written through a linear view by blits
            usage |= MTLTextureUsagePixelFormatView;
        }
        d.usage = usage;

        auto* img        = new NativeImage;
        img->format      = ci->format;
        img->type        = ci->imageType;
        img->extent      = ci->extent;
        img->mipLevels   = d.mipmapLevelCount;
        img->arrayLayers = layers;
        img->usage       = ci->usage;
        img->flags       = ci->flags;

        const MTLSizeAndAlign sa = [Device() heapTextureSizeAndAlignWithDescriptor:d];
        img->memorySize          = sa.size;
        img->memoryAlign         = sa.align;
        img->texture             = [Device() newTextureWithDescriptor:d];
        if( !img->texture )
        {
            delete img;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        *out = Handle< VkImage >( img );
        return VK_SUCCESS;
    }
}

void RTGL1::MetalNative::vkDestroyImage( VkDevice, VkImage image, const VkAllocationCallbacks* )
{
    auto* img = Obj< NativeImage >( image );
    if( img && !img->swapchain )
    {
        RTGL1::Metal::ForgetImage( image );
        delete img;
    }
}

VkResult RTGL1::MetalNative::vkCreateImageView( VkDevice,
                                                const VkImageViewCreateInfo* ci,
                                                const VkAllocationCallbacks*,
                                                VkImageView* out )
{
    @autoreleasepool
    {
        auto* img    = Obj< NativeImage >( ci->image );
        auto* v      = new NativeImageView;
        v->image     = img;
        v->format    = ci->format;
        *out         = Handle< VkImageView >( v );
        if( img->swapchain )
        {
            return VK_SUCCESS; // resolved to the drawable when used
        }

        const FormatInfo fi = GetFormatInfo( ci->format );
        bool             typeOk;
        const MTLTextureType type = TextureTypeOf( ci->viewType, typeOk );
        if( fi.mtl == MTLPixelFormatInvalid || !typeOk )
        {
            delete v;
            *out = VK_NULL_HANDLE;
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }

        const VkImageSubresourceRange& r = ci->subresourceRange;
        const uint32_t levels = r.levelCount == VK_REMAINING_MIP_LEVELS ? img->mipLevels - r.baseMipLevel : r.levelCount;
        const uint32_t layers =
            r.layerCount == VK_REMAINING_ARRAY_LAYERS ? img->arrayLayers - r.baseArrayLayer : r.layerCount;

        const MTLTextureSwizzleChannels sw = MTLTextureSwizzleChannelsMake(
            SwizzleOf( ci->components.r, MTLTextureSwizzleRed ),
            SwizzleOf( ci->components.g, MTLTextureSwizzleGreen ),
            SwizzleOf( ci->components.b, MTLTextureSwizzleBlue ),
            SwizzleOf( ci->components.a, MTLTextureSwizzleAlpha ) );
        const bool identity = sw.red == MTLTextureSwizzleRed && sw.green == MTLTextureSwizzleGreen &&
                              sw.blue == MTLTextureSwizzleBlue && sw.alpha == MTLTextureSwizzleAlpha;

        id< MTLTexture > base = img->texture;
        if( identity && fi.mtl == base.pixelFormat && type == base.textureType && r.baseMipLevel == 0 &&
            levels == base.mipmapLevelCount && r.baseArrayLayer == 0 && layers == img->arrayLayers )
        {
            v->texture = base;
        }
        else
        {
            const NSRange levelRange = NSMakeRange( r.baseMipLevel, levels );
            const NSRange sliceRange =
                img->type == VK_IMAGE_TYPE_3D ? NSMakeRange( 0, 1 ) : NSMakeRange( r.baseArrayLayer, layers );
            v->texture = identity ? [base newTextureViewWithPixelFormat:fi.mtl
                                                            textureType:type
                                                                 levels:levelRange
                                                                 slices:sliceRange]
                                  : [base newTextureViewWithPixelFormat:fi.mtl
                                                            textureType:type
                                                                 levels:levelRange
                                                                 slices:sliceRange
                                                                swizzle:sw];
        }
        if( !v->texture )
        {
            delete v;
            *out = VK_NULL_HANDLE;
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        return VK_SUCCESS;
    }
}

void RTGL1::MetalNative::vkDestroyImageView( VkDevice, VkImageView view, const VkAllocationCallbacks* )
{
    delete Obj< NativeImageView >( view );
}

// ================================================================================================
// objects whose state is kept by MetalBridge.mm (it records their create infos): handles only

#define RG_NATIVE_DUMMY_OBJECT( Type, Create, Destroy, CreateInfo )                              \
    VkResult RTGL1::MetalNative::Create( VkDevice, const CreateInfo*, const VkAllocationCallbacks*, Type* out ) \
    {                                                                                            \
        *out = Handle< Type >( new Dummy );                                                      \
        return VK_SUCCESS;                                                                       \
    }                                                                                            \
    void RTGL1::MetalNative::Destroy( VkDevice, Type h, const VkAllocationCallbacks* )           \
    {                                                                                            \
        delete Obj< Dummy >( h );                                                                \
    }

RG_NATIVE_DUMMY_OBJECT( VkSampler, vkCreateSampler, vkDestroySampler, VkSamplerCreateInfo )
RG_NATIVE_DUMMY_OBJECT( VkDescriptorSetLayout, vkCreateDescriptorSetLayout, vkDestroyDescriptorSetLayout, VkDescriptorSetLayoutCreateInfo )
RG_NATIVE_DUMMY_OBJECT( VkPipelineLayout, vkCreatePipelineLayout, vkDestroyPipelineLayout, VkPipelineLayoutCreateInfo )
RG_NATIVE_DUMMY_OBJECT( VkShaderModule, vkCreateShaderModule, vkDestroyShaderModule, VkShaderModuleCreateInfo )
RG_NATIVE_DUMMY_OBJECT( VkPipelineCache, vkCreatePipelineCache, vkDestroyPipelineCache, VkPipelineCacheCreateInfo )
RG_NATIVE_DUMMY_OBJECT( VkRenderPass, vkCreateRenderPass, vkDestroyRenderPass, VkRenderPassCreateInfo )
RG_NATIVE_DUMMY_OBJECT( VkFramebuffer, vkCreateFramebuffer, vkDestroyFramebuffer, VkFramebufferCreateInfo )

#undef RG_NATIVE_DUMMY_OBJECT

VkResult RTGL1::MetalNative::vkCreateComputePipelines( VkDevice,
                                                       VkPipelineCache,
                                                       uint32_t count,
                                                       const VkComputePipelineCreateInfo*,
                                                       const VkAllocationCallbacks*,
                                                       VkPipeline* out )
{
    for( uint32_t i = 0; i < count; i++ )
    {
        out[ i ] = Handle< VkPipeline >( new Dummy );
    }
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkCreateGraphicsPipelines( VkDevice,
                                                        VkPipelineCache,
                                                        uint32_t count,
                                                        const VkGraphicsPipelineCreateInfo*,
                                                        const VkAllocationCallbacks*,
                                                        VkPipeline* out )
{
    for( uint32_t i = 0; i < count; i++ )
    {
        out[ i ] = Handle< VkPipeline >( new Dummy );
    }
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyPipeline( VkDevice, VkPipeline p, const VkAllocationCallbacks* )
{
    delete Obj< Dummy >( p );
}

VkResult RTGL1::MetalNative::vkCreateDescriptorPool( VkDevice,
                                                     const VkDescriptorPoolCreateInfo*,
                                                     const VkAllocationCallbacks*,
                                                     VkDescriptorPool* out )
{
    *out = Handle< VkDescriptorPool >( new NativeDescriptorPool );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyDescriptorPool( VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* )
{
    RTGL1::MetalNative::vkResetDescriptorPool( device, pool, 0 );
    delete Obj< NativeDescriptorPool >( pool );
}

VkResult RTGL1::MetalNative::vkResetDescriptorPool( VkDevice, VkDescriptorPool pool, VkDescriptorPoolResetFlags )
{
    auto* p = Obj< NativeDescriptorPool >( pool );
    for( Dummy* d : p->sets )
    {
        delete d;
    }
    p->sets.clear();
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkAllocateDescriptorSets( VkDevice, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* out )
{
    auto* p = Obj< NativeDescriptorPool >( ai->descriptorPool );
    for( uint32_t i = 0; i < ai->descriptorSetCount; i++ )
    {
        auto* d = new Dummy;
        p->sets.push_back( d );
        out[ i ] = Handle< VkDescriptorSet >( d );
    }
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkFreeDescriptorSets( VkDevice, VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets )
{
    auto* p = Obj< NativeDescriptorPool >( pool );
    for( uint32_t i = 0; i < count; i++ )
    {
        auto* d = Obj< Dummy >( sets[ i ] );
        p->sets.erase( std::remove( p->sets.begin(), p->sets.end(), d ), p->sets.end() );
        delete d;
    }
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkUpdateDescriptorSets( VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const VkCopyDescriptorSet* )
{
    // the contents are recorded by MetalBridge.mm and encoded into argument buffers
}

// ================================================================================================
// synchronization

VkResult RTGL1::MetalNative::vkCreateFence( VkDevice, const VkFenceCreateInfo* ci, const VkAllocationCallbacks*, VkFence* out )
{
    auto sp     = std::make_shared< NativeFence >();
    auto* f     = sp.get();
    f->self     = std::move( sp );
    f->signaled = ( ci->flags & VK_FENCE_CREATE_SIGNALED_BIT ) != 0;
    *out        = Handle< VkFence >( f );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyFence( VkDevice, VkFence fence, const VkAllocationCallbacks* )
{
    if( auto* f = Obj< NativeFence >( fence ) )
    {
        // freed here unless a completion handler still holds it
        std::shared_ptr< NativeFence > drop = std::move( f->self );
    }
}

VkResult RTGL1::MetalNative::vkWaitForFences( VkDevice, uint32_t count, const VkFence* fences, VkBool32 waitAll, uint64_t timeoutNs )
{
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds( std::min< uint64_t >( timeoutNs, uint64_t( INT64_MAX / 2 ) ) );
    if( waitAll || count == 1 )
    {
        for( uint32_t i = 0; i < count; i++ )
        {
            auto*             f = Obj< NativeFence >( fences[ i ] );
            std::unique_lock  l( f->lock );
            if( !f->cv.wait_until( l, deadline, [ f ] { return f->signaled; } ) )
            {
                return VK_TIMEOUT;
            }
        }
        return VK_SUCCESS;
    }
    // any: poll
    while( true )
    {
        for( uint32_t i = 0; i < count; i++ )
        {
            auto*           f = Obj< NativeFence >( fences[ i ] );
            std::lock_guard l( f->lock );
            if( f->signaled )
            {
                return VK_SUCCESS;
            }
        }
        if( std::chrono::steady_clock::now() >= deadline )
        {
            return VK_TIMEOUT;
        }
        std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
    }
}

VkResult RTGL1::MetalNative::vkResetFences( VkDevice, uint32_t count, const VkFence* fences )
{
    for( uint32_t i = 0; i < count; i++ )
    {
        auto*           f = Obj< NativeFence >( fences[ i ] );
        std::lock_guard l( f->lock );
        f->signaled = false;
    }
    return VK_SUCCESS;
}

VkResult RTGL1::MetalNative::vkGetFenceStatus( VkDevice, VkFence fence )
{
    auto*           f = Obj< NativeFence >( fence );
    std::lock_guard l( f->lock );
    return f->signaled ? VK_SUCCESS : VK_NOT_READY;
}

VkResult RTGL1::MetalNative::vkCreateSemaphore( VkDevice, const VkSemaphoreCreateInfo* ci, const VkAllocationCallbacks*, VkSemaphore* out )
{
    auto* s = new NativeSemaphore;
    for( auto* n = static_cast< const VkBaseInStructure* >( ci->pNext ); n; n = n->pNext )
    {
        if( n->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO )
        {
            const auto* t = reinterpret_cast< const VkSemaphoreTypeCreateInfo* >( n );
            s->timeline   = t->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE;
            s->value      = t->initialValue;
        }
        else if( n->sType == VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT )
        {
            s->event = reinterpret_cast< const VkImportMetalSharedEventInfoEXT* >( n )->mtlSharedEvent;
        }
    }
    if( !s->event )
    {
        s->event               = [Device() newSharedEvent];
        s->event.signaledValue = s->value;
    }
    else
    {
        s->value = std::max< uint64_t >( s->value, s->event.signaledValue );
    }
    *out = Handle< VkSemaphore >( s );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroySemaphore( VkDevice, VkSemaphore semaphore, const VkAllocationCallbacks* )
{
    delete Obj< NativeSemaphore >( semaphore );
}

// ================================================================================================
// queries: no GPU timestamps (profiling marks in Metal measure command buffers instead)

VkResult RTGL1::MetalNative::vkCreateQueryPool( VkDevice, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks*, VkQueryPool* out )
{
    auto* q  = new NativeQueryPool;
    q->count = ci->queryCount;
    *out     = Handle< VkQueryPool >( q );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyQueryPool( VkDevice, VkQueryPool pool, const VkAllocationCallbacks* )
{
    delete Obj< NativeQueryPool >( pool );
}

VkResult RTGL1::MetalNative::vkGetQueryPoolResults( VkDevice,
                                                    VkQueryPool,
                                                    uint32_t,
                                                    uint32_t           count,
                                                    size_t             dataSize,
                                                    void*              data,
                                                    VkDeviceSize       stride,
                                                    VkQueryResultFlags flags )
{
    auto* p = static_cast< uint8_t* >( data );
    for( uint32_t i = 0; i < count && ( i * stride ) < dataSize; i++ )
    {
        uint8_t* q = p + i * stride;
        if( flags & VK_QUERY_RESULT_64_BIT )
        {
            reinterpret_cast< uint64_t* >( q )[ 0 ] = 0;
            if( flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT )
            {
                reinterpret_cast< uint64_t* >( q )[ 1 ] = 1;
            }
        }
        else
        {
            reinterpret_cast< uint32_t* >( q )[ 0 ] = 0;
            if( flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT )
            {
                reinterpret_cast< uint32_t* >( q )[ 1 ] = 1;
            }
        }
    }
    return VK_SUCCESS;
}

// ================================================================================================
// debug utils: nothing

VkResult RTGL1::MetalNative::vkCreateDebugUtilsMessengerEXT( VkInstance,
                                                             const VkDebugUtilsMessengerCreateInfoEXT*,
                                                             const VkAllocationCallbacks*,
                                                             VkDebugUtilsMessengerEXT* out )
{
    *out = Handle< VkDebugUtilsMessengerEXT >( new Dummy );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyDebugUtilsMessengerEXT( VkInstance, VkDebugUtilsMessengerEXT m, const VkAllocationCallbacks* )
{
    delete Obj< Dummy >( m );
}

VkResult RTGL1::MetalNative::vkSetDebugUtilsObjectNameEXT( VkDevice, const VkDebugUtilsObjectNameInfoEXT* )
{
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkCmdBeginDebugUtilsLabelEXT( VkCommandBuffer, const VkDebugUtilsLabelEXT* ) {}
void RTGL1::MetalNative::vkCmdEndDebugUtilsLabelEXT( VkCommandBuffer ) {}

// ================================================================================================
// acceleration structures: built by MetalBridge.mm (Metal::QueueBottomLevelBuilds / ...);
// the Vulkan objects only need an address

VkResult RTGL1::MetalNative::vkCreateAccelerationStructureKHR( VkDevice,
                                                               const VkAccelerationStructureCreateInfoKHR*,
                                                               const VkAllocationCallbacks*,
                                                               VkAccelerationStructureKHR* out )
{
    auto* a    = new NativeAccelerationStructure;
    a->address = G().nextASAddress.fetch_add( 1 << 16 );
    *out       = Handle< VkAccelerationStructureKHR >( a );
    return VK_SUCCESS;
}

void RTGL1::MetalNative::vkDestroyAccelerationStructureKHR( VkDevice, VkAccelerationStructureKHR as, const VkAllocationCallbacks* )
{
    delete Obj< NativeAccelerationStructure >( as );
}

VkDeviceAddress RTGL1::MetalNative::vkGetAccelerationStructureDeviceAddressKHR( VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR* info )
{
    auto* a = Obj< NativeAccelerationStructure >( info->accelerationStructure );
    return a ? a->address : 0;
}

void RTGL1::MetalNative::vkGetAccelerationStructureBuildSizesKHR( VkDevice,
                                                                  VkAccelerationStructureBuildTypeKHR,
                                                                  const VkAccelerationStructureBuildGeometryInfoKHR*,
                                                                  const uint32_t*,
                                                                  VkAccelerationStructureBuildSizesInfoKHR* sizes )
{
    // the Metal structures are allocated separately: the Vulkan side needs only small buffers
    sizes->accelerationStructureSize = 256;
    sizes->updateScratchSize         = 256;
    sizes->buildScratchSize          = 256;
}

void RTGL1::MetalNative::vkCmdBuildAccelerationStructuresKHR( VkCommandBuffer,
                                                              uint32_t,
                                                              const VkAccelerationStructureBuildGeometryInfoKHR*,
                                                              const VkAccelerationStructureBuildRangeInfoKHR* const* )
{
    Log( "vkCmdBuildAccelerationStructuresKHR isn't supported (builds go through the Metal path)" );
}

void RTGL1::MetalNative::vkCmdTraceRaysKHR( VkCommandBuffer,
                                            const VkStridedDeviceAddressRegionKHR*,
                                            const VkStridedDeviceAddressRegionKHR*,
                                            const VkStridedDeviceAddressRegionKHR*,
                                            const VkStridedDeviceAddressRegionKHR*,
                                            uint32_t,
                                            uint32_t,
                                            uint32_t )
{
    Log( "vkCmdTraceRaysKHR isn't supported" );
}

VkResult RTGL1::MetalNative::vkGetRayTracingShaderGroupHandlesKHR( VkDevice, VkPipeline, uint32_t, uint32_t, size_t, void* )
{
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult RTGL1::MetalNative::vkCreateRayTracingPipelinesKHR( VkDevice,
                                                             VkDeferredOperationKHR,
                                                             VkPipelineCache,
                                                             uint32_t,
                                                             const VkRayTracingPipelineCreateInfoKHR*,
                                                             const VkAllocationCallbacks*,
                                                             VkPipeline* )
{
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

// ================================================================================================
// VK_EXT_metal_objects

void RTGL1::MetalNative::vkExportMetalObjectsEXT( VkDevice, VkExportMetalObjectsInfoEXT* info )
{
    for( auto* n = static_cast< VkBaseOutStructure* >( const_cast< void* >( info->pNext ) ); n; n = n->pNext )
    {
        switch( n->sType )
        {
            case VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT:
                reinterpret_cast< VkExportMetalDeviceInfoEXT* >( n )->mtlDevice = Device();
                break;
            case VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT:
                reinterpret_cast< VkExportMetalCommandQueueInfoEXT* >( n )->mtlCommandQueue = Queue();
                break;
            case VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT: {
                auto* b      = reinterpret_cast< VkExportMetalBufferInfoEXT* >( n );
                auto* m      = Obj< NativeMemory >( b->memory );
                b->mtlBuffer = m ? EnsureMemoryBuffer( m ) : nil;
                break;
            }
            case VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT: {
                auto* t       = reinterpret_cast< VkExportMetalTextureInfoEXT* >( n );
                t->mtlTexture = nil;
                if( auto* v = Obj< NativeImageView >( t->imageView ) )
                {
                    t->mtlTexture = v->image && v->image->swapchain ? SwapchainTexture( Handle< VkImage >( v->image ) )
                                                                    : v->texture;
                }
                else if( auto* img = Obj< NativeImage >( t->image ) )
                {
                    t->mtlTexture = img->swapchain ? SwapchainTexture( t->image ) : img->texture;
                }
                break;
            }
            case VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT: {
                auto* e = reinterpret_cast< VkExportMetalSharedEventInfoEXT* >( n );
                if( auto* s = Obj< NativeSemaphore >( e->semaphore ) )
                {
                    e->mtlSharedEvent = s->event;
                }
                break;
            }
            default: break;
        }
    }
}
