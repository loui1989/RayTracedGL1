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

#define RG_METAL_BRIDGE_IMPLEMENTATION
#include "MetalBridge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

#include <algorithm>
#include <cfloat>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CommandBufferManager.h"
#include "MetalShaderGen.h"

using namespace RTGL1;

namespace
{

void Log( const char* fmt, ... ) __attribute__( ( format( printf, 1, 2 ) ) );
void Log( const char* fmt, ... )
{
    char    buf[ 2048 ];
    va_list args;
    va_start( args, fmt );
    vsnprintf( buf, sizeof( buf ), fmt, args );
    va_end( args );
    fprintf( stderr, "[Metal] %s\n", buf );
}

struct LayoutInfo
{
    std::vector< MetalGen::BindingDesc > bindings;
    MetalGen::SetSlots                   slots;
    bool                                 hasAccelerationStructure = false;
};

struct Descriptor
{
    VkImageView                view    = VK_NULL_HANDLE;
    VkSampler                  sampler = VK_NULL_HANDLE;
    VkBuffer                   buffer  = VK_NULL_HANDLE;
    VkDeviceSize               offset  = 0;
    VkAccelerationStructureKHR as      = VK_NULL_HANDLE;
};

// an encoded descriptor set; reused only when the GPU work that read it has completed
struct ArgBuffer
{
    id< MTLBuffer > buffer;
    uint64_t        lastUse = 0; // event value signaled when the last section using it is done
};
constexpr size_t MAX_ARG_BUFFERS_PER_SET = 32;

struct SetInfo
{
    std::shared_ptr< LayoutInfo >                          layout;
    VkDescriptorPool                                       pool = VK_NULL_HANDLE;
    std::unordered_map< uint32_t, std::vector< Descriptor > > descs;
    uint64_t                                               version = 1;

    std::vector< ArgBuffer > argBuffers;
    int                      argBufferCur        = -1;
    uint64_t                 encodedVersion      = 0;
    uint64_t                 encodedASGeneration = 0;
};

struct BufferInfo
{
    VkDeviceSize    size         = 0;
    VkDeviceMemory  memory       = VK_NULL_HANDLE;
    VkDeviceSize    memoryOffset = 0;
    VkDeviceAddress address      = 0;
};

struct SamplerInfo
{
    VkSamplerCreateInfo   ci = {};
    id< MTLSamplerState > mtl;
};

struct PipelineInfo
{
    VkShaderModule                          module = VK_NULL_HANDLE;
    std::string                             entry;
    VkPipelineLayout                        layout = VK_NULL_HANDLE;
    std::vector< VkSpecializationMapEntry > specEntries;
    std::vector< uint8_t >                  specData;

    id< MTLComputePipelineState > mtl;
    MTLSize                       threadsPerGroup = { 1, 1, 1 };
    std::string                   name;
    bool                          failed = false;
};

// what vkCmdBind* / vkCmdPushConstants set on SectionCmd()
constexpr uint32_t MAX_BOUND_SETS = 16;
struct BoundState
{
    VkPipeline      pipeline = VK_NULL_HANDLE;
    VkDescriptorSet sets[ MAX_BOUND_SETS ] = {};
    uint8_t         push[ 256 ]            = {};
    uint32_t        pushSize               = 0;
};

struct ProfiledCb
{
    id< MTLCommandBuffer > cb;
    const char*            name;
};

struct MtlAS
{
    id< MTLAccelerationStructure > as;
    NSUInteger                     size = 0;

    // top level: instance descriptors, written by the CPU for each build, so a buffer is
    // reused only after the GPU work that read it has completed
    std::vector< ArgBuffer > instances;

    // bottom level: the layout it was last built / refit with, to refit it instead of
    // rebuilding when a per-frame rebuilt (dynamic) structure keeps the same layout
    std::vector< uint64_t > layout;
    uint32_t                refitsSinceBuild = 0;
};

// ---- graphics shadow state

struct RenderPassInfo
{
    struct Attachment
    {
        VkFormat            format = VK_FORMAT_UNDEFINED;
        VkAttachmentLoadOp  load   = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkAttachmentStoreOp store  = VK_ATTACHMENT_STORE_OP_STORE;
    };
    std::vector< Attachment > attachments;
    std::vector< uint32_t >   colorRefs; // attachment index or VK_ATTACHMENT_UNUSED
    uint32_t                  depthRef = VK_ATTACHMENT_UNUSED;
    uint32_t                  viewMask = 0;
    bool                      supported = true;
};

struct FramebufferInfo
{
    VkRenderPass               renderPass = VK_NULL_HANDLE;
    std::vector< VkImageView > views;
    uint32_t                   width = 0, height = 0, layers = 1;
};

struct ShaderStageInfo
{
    VkShaderModule                          module = VK_NULL_HANDLE;
    std::string                             entry;
    std::vector< VkSpecializationMapEntry > specEntries;
    std::vector< uint8_t >                  specData;
};

struct GraphicsPipelineInfo
{
    ShaderStageInfo vs, fs;
    bool            hasFs = false;

    std::vector< VkVertexInputBindingDescription >   bindings;
    std::vector< VkVertexInputAttributeDescription > attributes;
    VkPrimitiveTopology                              topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkCullModeFlags cullMode   = VK_CULL_MODE_NONE;
    VkFrontFace     frontFace  = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool            depthBias  = false;
    float           depthBiasConstant = 0, depthBiasSlope = 0, depthBiasClamp = 0;
    bool            wireframe  = false;

    bool        depthTest    = false;
    bool        depthWrite   = false;
    VkCompareOp depthCompare = VK_COMPARE_OP_ALWAYS;

    std::vector< VkPipelineColorBlendAttachmentState > blends;

    bool       dynamicViewport = false, dynamicScissor = false;
    VkViewport viewport = {};
    VkRect2D   scissor  = {};

    VkPipelineLayout layout     = VK_NULL_HANDLE;
    VkRenderPass     renderPass = VK_NULL_HANDLE;
    uint32_t         viewCount  = 1;
    std::string      name;

    // Metal
    bool                                                          failed = false;
    id< MTLFunction >                                             vfn, ffn;
    std::unordered_map< std::string, id< MTLRenderPipelineState > > variants; // by attachment formats
    id< MTLDepthStencilState >                                    depthState;
};

struct BlasBuild
{
    VkAccelerationStructureKHR                              dst   = VK_NULL_HANDLE;
    VkBuildAccelerationStructureFlagsKHR                    flags = 0;
    bool                                                    update = false;
    std::vector< VkAccelerationStructureGeometryKHR >       geoms;
    std::vector< VkAccelerationStructureBuildRangeInfoKHR > ranges;
};

struct TlasBuild
{
    VkAccelerationStructureKHR                        dst       = VK_NULL_HANDLE;
    bool                                              fastTrace = true;
    std::vector< VkAccelerationStructureInstanceKHR > instances;
};

struct State
{
    std::recursive_mutex lock;

    // shadow of the Vulkan objects
    std::unordered_map< VkDescriptorSetLayout, std::shared_ptr< LayoutInfo > >          layouts;
    std::unordered_map< VkPipelineLayout, std::vector< std::shared_ptr< LayoutInfo > > > pipelineLayouts;
    std::unordered_map< VkDescriptorSet, SetInfo >                                      sets;
    std::unordered_map< VkSampler, SamplerInfo >                                        samplers;
    std::unordered_map< VkBuffer, BufferInfo >                                          buffers;
    std::map< VkDeviceAddress, VkBuffer >                                               buffersByAddress;
    std::unordered_map< VkDeviceMemory, id< MTLBuffer > >                               memoryBuffers;
    std::unordered_map< VkImageView, id< MTLTexture > >                                 viewTextures;
    std::unordered_map< VkImage, id< MTLTexture > >                                     imageTextures;
    std::unordered_map< VkShaderModule, std::vector< uint32_t > >                       modules;
    std::unordered_map< VkPipeline, PipelineInfo >                                      pipelines;

    // native path
    bool                                    enabled  = false;
    VkDevice                                device   = VK_NULL_HANDLE;
    PFN_vkExportMetalObjectsEXT             exportFn = nullptr;
    std::shared_ptr< CommandBufferManager > cmdManager;

    id< MTLDevice >       mtlDevice;
    id< MTLCommandQueue > queue;
    id< MTLSharedEvent >  event;
    VkSemaphore           semaphore = VK_NULL_HANDLE;
    uint64_t              timeline  = 0;
    id< MTLFence >        blasFence; // bottom level builds -> top level builds
    id< MTLFence >        fence;     // acceleration structure builds -> path tracing

    id                                    residency; // id<MTLResidencySet>, macOS 15+
    std::unordered_map< void*, uint32_t > resident;  // reference counts (views can share a texture)
    bool                       residencyDirty = false;

    std::unordered_map< VkAccelerationStructureKHR, MtlAS >     as;
    std::unordered_map< VkDeviceAddress, VkAccelerationStructureKHR > asByAddress;
    uint64_t                                                    asGeneration = 1;
    std::vector< BlasBuild >                                    pendingBlas;
    std::vector< TlasBuild >                                    pendingTlas;
    id< MTLBuffer >                                             scratch;
    id< MTLAccelerationStructure >                              dummyBlas;

    id< MTLCommandBuffer >         cb;
    id< MTLComputeCommandEncoder > encoder;
    id< MTLCommandBuffer >         lastCb;
    uint64_t                       sectionDone = 0; // event value the current section signals
    BoundState                     bound;

    // graphics
    std::unordered_map< VkRenderPass, RenderPassInfo >        renderPasses;
    std::unordered_map< VkFramebuffer, FramebufferInfo >      framebuffers;
    std::unordered_map< VkPipeline, GraphicsPipelineInfo >    gpipelines;
    bool                                                      rasterEnv    = true;
    bool                                                      rasterBroken = false;
    id< MTLComputePipelineState >                             compactIndirect;
    id< MTLComputePipelineState >                             blitPipeline;
    id< MTLSamplerState >                                     blitNearest, blitLinear;
    id< MTLBuffer >                                           indirectScratch;

    // section encoders: one at a time, ordered with passFence
    id< MTLRenderCommandEncoder > renderEnc;
    id< MTLBlitCommandEncoder >   blitEnc;
    bool                          waitASFence = false;
    bool                           computeEnv    = true;
    bool                           fastMath      = false; // RTGL1_METAL_FASTMATH=1
    uint32_t                       asRebuildEvery = 8;    // RTGL1_METAL_AS_REBUILD_EVERY, 1 = never refit
    struct
    {
        uint32_t blasBuilds = 0, blasRefits = 0, tlasInstances = 0;
    } asStats;
    bool                           computeBroken = false;

    // RTGL1_PROFILE=1: a section is split into a command buffer per mark
    bool                              profile = false;
    id< MTLFence >                    passFence;
    id< MTLCommandBuffer >            profWaitCb;
    std::vector< ProfiledCb >         profCbs;
    std::unordered_set< std::string > internedNames;
    std::mutex                        timingLock;
    std::vector< std::pair< const char*, double > > timings;
    std::atomic< uint32_t >        loggedErrors{ 0 };
    std::unordered_set< std::string > loggedOnce;
};

State& S()
{
    static State s;
    return s;
}

void LogOnce( const std::string& msg )
{
    if( S().loggedOnce.insert( msg ).second )
    {
        Log( "%s", msg.c_str() );
    }
}

// ---- residency

void MakeResident( id< MTLResource > r )
{
    State& s = S();
    if( !r || !s.residency )
    {
        return;
    }
    void* key = ( __bridge void* )r;
    if( ++s.resident[ key ] == 1 )
    {
        if( @available( macOS 15.0, * ) )
        {
            [( id< MTLResidencySet > )s.residency addAllocation:( id< MTLAllocation > )r];
            s.residencyDirty = true;
        }
    }
}

void MakeResidentAS( id< MTLAccelerationStructure > r )
{
    MakeResident( ( id< MTLResource > )r );
}

void RemoveResident( id< MTLResource > r )
{
    State& s = S();
    if( !r || !s.residency )
    {
        return;
    }
    auto it = s.resident.find( ( __bridge void* )r );
    if( it == s.resident.end() )
    {
        return;
    }
    if( --it->second == 0 )
    {
        s.resident.erase( it );
        if( @available( macOS 15.0, * ) )
        {
            [( id< MTLResidencySet > )s.residency removeAllocation:( id< MTLAllocation > )r];
            s.residencyDirty = true;
        }
    }
}

// ---- exports from MoltenVK

id< MTLTexture > TextureForImage( VkImage image )
{
    State& s = S();
    if( image == VK_NULL_HANDLE || !s.exportFn )
    {
        return nil;
    }
    auto it = s.imageTextures.find( image );
    if( it != s.imageTextures.end() )
    {
        return it->second;
    }

    VkExportMetalTextureInfoEXT t = { VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT };
    t.image                       = image;
    t.plane                       = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT info = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
    info.pNext                       = &t;
    s.exportFn( s.device, &info );

    id< MTLTexture > tex = t.mtlTexture;
    if( !tex )
    {
        LogOnce( "couldn't get the Metal texture of an image" );
        return nil;
    }
    s.imageTextures[ image ] = tex;
    MakeResident( tex );
    return tex;
}

id< MTLTexture > TextureForView( VkImageView view )
{
    State& s = S();
    if( view == VK_NULL_HANDLE || !s.exportFn )
    {
        return nil;
    }
    auto it = s.viewTextures.find( view );
    if( it != s.viewTextures.end() )
    {
        return it->second;
    }

    VkExportMetalTextureInfoEXT t = { VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT };
    t.imageView                   = view;
    t.plane                       = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT info = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
    info.pNext                       = &t;
    s.exportFn( s.device, &info );

    id< MTLTexture > tex = t.mtlTexture;
    if( !tex )
    {
        LogOnce( "couldn't get the Metal texture of an image view" );
        return nil;
    }
    s.viewTextures[ view ] = tex;
    MakeResident( tex );
    if( tex.parentTexture )
    {
        MakeResident( tex.parentTexture );
    }
    return tex;
}

id< MTLBuffer > BufferForMemory( VkDeviceMemory memory )
{
    State& s = S();
    if( memory == VK_NULL_HANDLE || !s.exportFn )
    {
        return nil;
    }
    auto it = s.memoryBuffers.find( memory );
    if( it != s.memoryBuffers.end() )
    {
        return it->second;
    }

    VkExportMetalBufferInfoEXT b = { VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT };
    b.memory                     = memory;
    VkExportMetalObjectsInfoEXT info = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
    info.pNext                       = &b;
    s.exportFn( s.device, &info );

    id< MTLBuffer > buf = b.mtlBuffer;
    if( !buf )
    {
        LogOnce( "couldn't get the Metal buffer of a memory allocation" );
        return nil;
    }
    s.memoryBuffers[ memory ] = buf;
    MakeResident( buf );
    return buf;
}

// Metal buffer and byte offset of a Vulkan buffer (+ offset)
bool ResolveBuffer( VkBuffer buffer, VkDeviceSize offset, id< MTLBuffer > __strong& outBuf, NSUInteger& outOffset )
{
    State& s  = S();
    auto   it = s.buffers.find( buffer );
    if( it == s.buffers.end() || it->second.memory == VK_NULL_HANDLE )
    {
        return false;
    }
    outBuf    = BufferForMemory( it->second.memory );
    outOffset = NSUInteger( it->second.memoryOffset + offset );
    return outBuf != nil;
}

// Metal buffer and byte offset of a Vulkan buffer device address
bool ResolveAddress( VkDeviceAddress address, id< MTLBuffer > __strong& outBuf, NSUInteger& outOffset )
{
    State& s  = S();
    auto   it = s.buffersByAddress.upper_bound( address );
    if( it == s.buffersByAddress.begin() )
    {
        return false;
    }
    --it;
    const VkDeviceAddress base = it->first;
    auto                  bi   = s.buffers.find( it->second );
    if( bi == s.buffers.end() || address >= base + std::max< VkDeviceSize >( bi->second.size, 1 ) )
    {
        return false;
    }
    return ResolveBuffer( it->second, address - base, outBuf, outOffset );
}

uint64_t BufferGPUAddress( VkBuffer buffer, VkDeviceSize offset )
{
    id< MTLBuffer > b;
    NSUInteger      o = 0;
    if( !ResolveBuffer( buffer, offset, b, o ) )
    {
        return 0;
    }
    return b.gpuAddress + o;
}

// ---- samplers

MTLSamplerAddressMode AddressMode( VkSamplerAddressMode m )
{
    switch( m )
    {
        case VK_SAMPLER_ADDRESS_MODE_REPEAT: return MTLSamplerAddressModeRepeat;
        case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
        case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
        case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: return MTLSamplerAddressModeMirrorClampToEdge;
        case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
        default: return MTLSamplerAddressModeClampToEdge;
    }
}

MTLCompareFunction CompareFunction( VkCompareOp op )
{
    switch( op )
    {
        case VK_COMPARE_OP_NEVER: return MTLCompareFunctionNever;
        case VK_COMPARE_OP_LESS: return MTLCompareFunctionLess;
        case VK_COMPARE_OP_EQUAL: return MTLCompareFunctionEqual;
        case VK_COMPARE_OP_LESS_OR_EQUAL: return MTLCompareFunctionLessEqual;
        case VK_COMPARE_OP_GREATER: return MTLCompareFunctionGreater;
        case VK_COMPARE_OP_NOT_EQUAL: return MTLCompareFunctionNotEqual;
        case VK_COMPARE_OP_GREATER_OR_EQUAL: return MTLCompareFunctionGreaterEqual;
        case VK_COMPARE_OP_ALWAYS:
        default: return MTLCompareFunctionAlways;
    }
}

id< MTLSamplerState > SamplerFor( VkSampler sampler )
{
    State& s  = S();
    auto   it = s.samplers.find( sampler );
    if( it == s.samplers.end() || !s.mtlDevice )
    {
        return nil;
    }
    if( it->second.mtl )
    {
        return it->second.mtl;
    }

    const VkSamplerCreateInfo& ci = it->second.ci;
    MTLSamplerDescriptor*      d  = [MTLSamplerDescriptor new];
    d.minFilter    = ci.minFilter == VK_FILTER_NEAREST ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    d.magFilter    = ci.magFilter == VK_FILTER_NEAREST ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    d.mipFilter    = ci.mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    d.sAddressMode = AddressMode( ci.addressModeU );
    d.tAddressMode = AddressMode( ci.addressModeV );
    d.rAddressMode = AddressMode( ci.addressModeW );
    d.maxAnisotropy =
        ci.anisotropyEnable ? NSUInteger( std::clamp( ci.maxAnisotropy, 1.0f, 16.0f ) ) : 1;
    d.lodMinClamp           = ci.minLod;
    d.lodMaxClamp           = ci.maxLod >= VK_LOD_CLAMP_NONE ? FLT_MAX : ci.maxLod;
    d.normalizedCoordinates = !ci.unnormalizedCoordinates;
    if( ci.compareEnable )
    {
        d.compareFunction = CompareFunction( ci.compareOp );
    }
    switch( ci.borderColor )
    {
        case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
        case VK_BORDER_COLOR_INT_OPAQUE_BLACK: d.borderColor = MTLSamplerBorderColorOpaqueBlack; break;
        case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
        case VK_BORDER_COLOR_INT_OPAQUE_WHITE: d.borderColor = MTLSamplerBorderColorOpaqueWhite; break;
        default: d.borderColor = MTLSamplerBorderColorTransparentBlack; break;
    }
    d.supportArgumentBuffers = YES;

    it->second.mtl = [s.mtlDevice newSamplerStateWithDescriptor:d];
    return it->second.mtl;
}

// ---- argument buffers

void WriteID( uint64_t* dst, uint32_t slot, MTLResourceID rid )
{
    static_assert( sizeof( MTLResourceID ) == sizeof( uint64_t ) );
    memcpy( &dst[ slot ], &rid, sizeof( uint64_t ) );
}

id< MTLBuffer > EncodeSet( VkDescriptorSet set )
{
    State& s  = S();
    auto   it = s.sets.find( set );
    if( it == s.sets.end() || !it->second.layout )
    {
        LogOnce( "a descriptor set used by the native path is unknown" );
        return nil;
    }
    SetInfo&          si     = it->second;
    const LayoutInfo& layout = *si.layout;

    const bool upToDate = si.encodedVersion == si.version &&
                          ( !layout.hasAccelerationStructure || si.encodedASGeneration == s.asGeneration );
    if( upToDate && si.argBufferCur >= 0 )
    {
        ArgBuffer& cur = si.argBuffers[ si.argBufferCur ];
        cur.lastUse    = s.sectionDone;
        return cur.buffer;
    }

    // re-encode into a buffer that no submitted GPU work reads anymore
    const uint64_t completed = s.event.signaledValue;
    int            pick      = -1;
    for( size_t i = 0; i < si.argBuffers.size(); i++ )
    {
        if( si.argBuffers[ i ].lastUse <= completed && int( i ) != si.argBufferCur )
        {
            pick = int( i );
            break;
        }
    }
    if( pick < 0 )
    {
        if( si.argBuffers.size() < MAX_ARG_BUFFERS_PER_SET )
        {
            si.argBuffers.push_back( {} );
            pick = int( si.argBuffers.size() - 1 );
        }
        else
        {
            LogOnce( "a descriptor set changes too often, its Metal copy may be overwritten while in use" );
            pick = ( si.argBufferCur + 1 ) % int( si.argBuffers.size() );
        }
    }
    si.argBufferCur        = pick;
    si.argBuffers[ pick ].lastUse = s.sectionDone;

    const NSUInteger bytes = std::max< NSUInteger >( 1, layout.slots.totalSlots ) * sizeof( uint64_t );
    id< MTLBuffer > __strong& ab    = si.argBuffers[ pick ].buffer;
    if( !ab || ab.length < bytes )
    {
        ab = [s.mtlDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        ab.label = @"RTGL1 descriptor set";
    }

    auto* dst = static_cast< uint64_t* >( ab.contents );
    memset( dst, 0, bytes );

    for( const MetalGen::SlotInfo& slot : layout.slots.slots )
    {
        auto        dit   = si.descs.find( slot.binding );
        const auto* descs = dit != si.descs.end() ? &dit->second : nullptr;

        for( uint32_t i = 0; i < slot.count; i++ )
        {
            if( !descs || i >= descs->size() )
            {
                break;
            }
            const Descriptor& d = ( *descs )[ i ];

            switch( slot.type )
            {
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    if( id< MTLTexture > t = TextureForView( d.view ) )
                    {
                        WriteID( dst, slot.textureSlot + i, t.gpuResourceID );
                    }
                    break;

                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    if( id< MTLTexture > t = TextureForView( d.view ) )
                    {
                        WriteID( dst, slot.textureSlot + i, t.gpuResourceID );
                    }
                    if( id< MTLSamplerState > smp = SamplerFor( d.sampler ) )
                    {
                        WriteID( dst, slot.samplerSlot + i, smp.gpuResourceID );
                    }
                    break;

                case VK_DESCRIPTOR_TYPE_SAMPLER:
                    if( id< MTLSamplerState > smp = SamplerFor( d.sampler ) )
                    {
                        WriteID( dst, slot.samplerSlot + i, smp.gpuResourceID );
                    }
                    break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    dst[ slot.bufferSlot + i ] = BufferGPUAddress( d.buffer, d.offset );
                    break;

                case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
                    auto a = s.as.find( d.as );
                    if( a != s.as.end() && a->second.as )
                    {
                        WriteID( dst, slot.bufferSlot + i, a->second.as.gpuResourceID );
                    }
                    break;
                }

                default: LogOnce( "unsupported descriptor type " + std::to_string( slot.type ) ); break;
            }
        }
    }

    si.encodedVersion      = si.version;
    si.encodedASGeneration = s.asGeneration;
    return ab;
}

// ---- acceleration structures

MTLAccelerationStructureUsage UsageOf( VkBuildAccelerationStructureFlagsKHR flags )
{
    MTLAccelerationStructureUsage u = MTLAccelerationStructureUsageNone;
    if( flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR )
    {
        u |= MTLAccelerationStructureUsageRefit;
    }
    if( flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR )
    {
        u |= MTLAccelerationStructureUsagePreferFastBuild;
    }
    return u;
}

MTLPrimitiveAccelerationStructureDescriptor* MakePrimitiveDescriptor( const BlasBuild& b )
{
    NSMutableArray* geoms = [NSMutableArray arrayWithCapacity:b.geoms.size()];

    for( size_t i = 0; i < b.geoms.size(); i++ )
    {
        const VkAccelerationStructureGeometryKHR&       g = b.geoms[ i ];
        const VkAccelerationStructureBuildRangeInfoKHR& r = b.ranges[ i ];
        if( g.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR )
        {
            LogOnce( "only triangle geometry is supported" );
            return nil;
        }
        const auto& t = g.geometry.triangles;
        if( t.vertexFormat != VK_FORMAT_R32G32B32_SFLOAT )
        {
            LogOnce( "unsupported vertex format " + std::to_string( t.vertexFormat ) );
            return nil;
        }

        const bool indexed = t.indexType != VK_INDEX_TYPE_NONE_KHR;

        MTLAccelerationStructureTriangleGeometryDescriptor* tri =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];

        // Vulkan: with indices, primitiveOffset is into the index data and firstVertex is added
        // to the indices; without, both offset the vertex data
        const VkDeviceAddress vertexAddress =
            t.vertexData.deviceAddress + VkDeviceSize( r.firstVertex ) * t.vertexStride +
            ( indexed ? 0 : r.primitiveOffset );

        id< MTLBuffer > vb;
        NSUInteger      vo = 0;
        if( !ResolveAddress( vertexAddress, vb, vo ) )
        {
            LogOnce( "couldn't find the Metal buffer of acceleration structure vertices" );
            return nil;
        }
        tri.vertexBuffer       = vb;
        tri.vertexBufferOffset = vo;
        tri.vertexStride       = t.vertexStride;
        tri.vertexFormat       = MTLAttributeFormatFloat3;

        if( indexed )
        {
            id< MTLBuffer > ib;
            NSUInteger      io = 0;
            if( !ResolveAddress( t.indexData.deviceAddress + r.primitiveOffset, ib, io ) )
            {
                LogOnce( "couldn't find the Metal buffer of acceleration structure indices" );
                return nil;
            }
            tri.indexBuffer       = ib;
            tri.indexBufferOffset = io;
            tri.indexType = t.indexType == VK_INDEX_TYPE_UINT16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        }

        if( t.transformData.deviceAddress != 0 )
        {
            id< MTLBuffer > tb;
            NSUInteger      to = 0;
            if( !ResolveAddress( t.transformData.deviceAddress + r.transformOffset, tb, to ) )
            {
                LogOnce( "couldn't find the Metal buffer of acceleration structure transforms" );
                return nil;
            }
            tri.transformationMatrixBuffer       = tb;
            tri.transformationMatrixBufferOffset = to;
            if( @available( macOS 15.0, * ) )
            {
                // VkTransformMatrixKHR is a row-major 3x4 matrix
                tri.transformationMatrixLayout = MTLMatrixLayoutRowMajor;
            }
        }

        tri.triangleCount = r.primitiveCount;
        tri.opaque        = ( g.flags & VK_GEOMETRY_OPAQUE_BIT_KHR ) != 0;
        tri.allowDuplicateIntersectionFunctionInvocation =
            ( g.flags & VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR ) == 0;

        [geoms addObject:tri];
    }

    MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    d.geometryDescriptors                          = geoms;
    d.usage                                        = UsageOf( b.flags );
    return d;
}

// Makes 'm' at least 'size' bytes; returns true if the acceleration structure was recreated.
bool EnsureAS( MtlAS& m, NSUInteger size )
{
    State& s = S();
    if( m.as && m.size >= size )
    {
        return false;
    }
    if( m.as )
    {
        RemoveResident( ( id< MTLResource > )m.as );
    }
    // some headroom, as dynamic geometry grows and shrinks every frame
    const NSUInteger alloc = size + size / 4;
    m.as                   = [s.mtlDevice newAccelerationStructureWithSize:alloc];
    m.size                 = m.as ? alloc : 0;
    if( !m.as )
    {
        LogOnce( "couldn't allocate an acceleration structure" );
        return true;
    }
    MakeResidentAS( m.as );
    s.asGeneration++;
    return true;
}

void EnsureScratch( NSUInteger size )
{
    State& s = S();
    if( s.scratch && s.scratch.length >= size )
    {
        return;
    }
    s.scratch       = [s.mtlDevice newBufferWithLength:std::max< NSUInteger >( size + size / 4, 1 << 20 )
                                         options:MTLResourceStorageModePrivate];
    s.scratch.label = @"RTGL1 acceleration structure scratch";
}

NSUInteger Align256( NSUInteger x )
{
    return ( x + 255 ) & ~NSUInteger( 255 );
}

// A tiny placeholder bottom level structure, so a top level structure can always be built
// (also with no instances) and the shaders never see a missing one.
bool EnsureDummyBlas( id< MTLAccelerationStructureCommandEncoder > ae, NSUInteger& scratchOffset )
{
    State& s = S();
    if( s.dummyBlas )
    {
        return true;
    }
    const float verts[ 9 ] = { 0, -1e6f, 0, 1e-3f, -1e6f, 0, 0, -1e6f, 1e-3f };
    id< MTLBuffer > vb     = [s.mtlDevice newBufferWithBytes:verts length:sizeof( verts ) options:MTLResourceStorageModeShared];

    MTLAccelerationStructureTriangleGeometryDescriptor* tri =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    tri.vertexBuffer  = vb;
    tri.vertexStride  = 3 * sizeof( float );
    tri.vertexFormat  = MTLAttributeFormatFloat3;
    tri.triangleCount = 1;
    tri.opaque        = YES;

    MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    d.geometryDescriptors                          = @[ tri ];

    const MTLAccelerationStructureSizes sizes = [s.mtlDevice accelerationStructureSizesWithDescriptor:d];
    s.dummyBlas = [s.mtlDevice newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    if( !s.dummyBlas )
    {
        return false;
    }
    MakeResidentAS( s.dummyBlas );
    EnsureScratch( scratchOffset + Align256( std::max< NSUInteger >( sizes.buildScratchBufferSize, 1 ) ) );
    [ae buildAccelerationStructure:s.dummyBlas descriptor:d scratchBuffer:s.scratch scratchBufferOffset:scratchOffset];
    scratchOffset += Align256( std::max< NSUInteger >( sizes.buildScratchBufferSize, 1 ) );
    return true;
}

void        CommitResidency();
const char* Intern( const char* name );

// Encodes the queued builds into s.cb. Returns true if s.fence is updated after them.
bool EncodeAccelerationStructureBuilds()
{
    State& s = S();
    if( s.pendingBlas.empty() && s.pendingTlas.empty() )
    {
        return false;
    }

    bool blasEncoded = false;

    // bottom level (and the placeholder, if the top level needs it)
    if( !s.pendingBlas.empty() || ( !s.pendingTlas.empty() && !s.dummyBlas ) )
    {
        struct Item
        {
            BlasBuild*                                   build;
            MTLPrimitiveAccelerationStructureDescriptor* desc;
            bool                                         refit;
            NSUInteger                                   scratchOffset;
        };
        std::vector< Item > items;
        NSUInteger          scratchTotal = 0;

        for( BlasBuild& b : s.pendingBlas )
        {
            MTLPrimitiveAccelerationStructureDescriptor* d = MakePrimitiveDescriptor( b );
            if( !d )
            {
                continue;
            }
            // Geometry rebuilt every frame (dynamic, built with "prefer fast build"): refit
            // the previous structure instead when the layout is the same, with a full build
            // every asRebuildEvery frames to restore the quality of the hierarchy
            const bool perFrame =
                !b.update && ( b.flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR ) &&
                s.asRebuildEvery > 1;
            std::vector< uint64_t > layout;
            if( perFrame )
            {
                d.usage |= MTLAccelerationStructureUsageRefit;
                for( size_t g = 0; g < b.geoms.size(); g++ )
                {
                    const auto& t = b.geoms[ g ].geometry.triangles;
                    layout.push_back( ( uint64_t( b.ranges[ g ].primitiveCount ) << 32 ) |
                                      ( uint64_t( t.indexType ) << 8 ) | uint64_t( b.geoms[ g ].flags ) );
                }
            }

            const MTLAccelerationStructureSizes sizes = [s.mtlDevice accelerationStructureSizesWithDescriptor:d];

            MtlAS&     m         = s.as[ b.dst ];
            const bool recreated = EnsureAS( m, sizes.accelerationStructureSize );
            if( !m.as )
            {
                continue;
            }
            bool refit = b.update && !recreated && ( d.usage & MTLAccelerationStructureUsageRefit );
            if( perFrame )
            {
                refit = !recreated && m.layout == layout && m.refitsSinceBuild + 1 < s.asRebuildEvery;
                m.refitsSinceBuild = refit ? m.refitsSinceBuild + 1 : 0;
                m.layout           = std::move( layout );
            }
            refit ? s.asStats.blasRefits++ : s.asStats.blasBuilds++;

            items.push_back( { &b, d, refit, scratchTotal } );
            scratchTotal += Align256( std::max< NSUInteger >(
                refit ? sizes.refitScratchBufferSize : sizes.buildScratchBufferSize, 1 ) );
        }

        EnsureScratch( std::max< NSUInteger >( scratchTotal, 1 ) );

        id< MTLAccelerationStructureCommandEncoder > ae = [s.cb accelerationStructureCommandEncoder];
        ae.label                                        = @"RTGL1 bottom level";
        for( const Item& it : items )
        {
            MtlAS& m = s.as[ it.build->dst ];
            if( it.refit )
            {
                [ae refitAccelerationStructure:m.as
                                    descriptor:it.desc
                                   destination:nil
                                 scratchBuffer:s.scratch
                           scratchBufferOffset:it.scratchOffset];
            }
            else
            {
                [ae buildAccelerationStructure:m.as
                                    descriptor:it.desc
                                 scratchBuffer:s.scratch
                           scratchBufferOffset:it.scratchOffset];
            }
        }
        if( !s.pendingTlas.empty() )
        {
            NSUInteger off = scratchTotal;
            EnsureDummyBlas( ae, off );
        }
        // the top level build waits for these; without one, the path tracing does
        [ae updateFence:s.pendingTlas.empty() ? s.fence : s.blasFence];
        [ae endEncoding];
        blasEncoded = true;
        s.pendingBlas.clear();

        if( s.profile && !s.pendingTlas.empty() )
        {
            // measured separately from the top level build
            CommitResidency();
            [s.cb commit];
            s.profCbs.push_back( { s.cb, Intern( "AS bottom level" ) } );
            s.cb       = [s.queue commandBuffer];
            s.cb.label = @"RTGL1 native (profiled)";
        }
    }

    if( s.pendingTlas.empty() )
    {
        return blasEncoded;
    }

    // top level
    id< MTLAccelerationStructureCommandEncoder > ae = [s.cb accelerationStructureCommandEncoder];
    ae.label                                        = @"RTGL1 top level";
    if( blasEncoded )
    {
        [ae waitForFence:s.blasFence];
    }

    NSUInteger scratchOffset = 0;
    for( TlasBuild& t : s.pendingTlas )
    {
        // the bottom level structures, as Metal references them by index
        NSMutableArray< id< MTLAccelerationStructure > >* blasList = [NSMutableArray array];
        std::unordered_map< VkAccelerationStructureKHR, uint32_t > blasIndex;

        // at least one instance, so that there is always a valid structure
        const NSUInteger count = std::max< NSUInteger >( t.instances.size(), 1 );
        MtlAS&           m     = s.as[ t.dst ];
        const NSUInteger instBytes = count * sizeof( MTLAccelerationStructureUserIDInstanceDescriptor );
        id< MTLBuffer >  instBuf;
        {
            const uint64_t completed = s.event.signaledValue;
            ArgBuffer*     slot      = nullptr;
            for( ArgBuffer& a : m.instances )
            {
                if( a.lastUse <= completed )
                {
                    slot = &a;
                    break;
                }
            }
            if( !slot )
            {
                m.instances.push_back( {} );
                slot = &m.instances.back();
            }
            if( !slot->buffer || slot->buffer.length < instBytes )
            {
                slot->buffer       = [s.mtlDevice newBufferWithLength:instBytes + instBytes / 4
                                                        options:MTLResourceStorageModeShared];
                slot->buffer.label = @"RTGL1 TLAS instances";
            }
            slot->lastUse = s.sectionDone;
            instBuf       = slot->buffer;
        }
        s.asStats.tlasInstances += uint32_t( t.instances.size() );
        auto* dst = static_cast< MTLAccelerationStructureUserIDInstanceDescriptor* >( instBuf.contents );

        // index 0: the placeholder, for instances that are hidden
        if( s.dummyBlas )
        {
            [blasList addObject:s.dummyBlas];
        }

        for( NSUInteger i = 0; i < count; i++ )
        {
            MTLAccelerationStructureUserIDInstanceDescriptor& d = dst[ i ];
            d                                                    = {};
            d.mask                                               = 0;
            d.accelerationStructureIndex                         = 0;
            for( int c = 0; c < 4; c++ )
            {
                for( int r = 0; r < 3; r++ )
                {
                    d.transformationMatrix.columns[ c ].elements[ r ] = ( r == c ) ? 1.0f : 0.0f;
                }
            }
            if( i >= t.instances.size() )
            {
                continue;
            }

            const VkAccelerationStructureInstanceKHR& src = t.instances[ i ];
            for( int c = 0; c < 4; c++ )
            {
                for( int r = 0; r < 3; r++ )
                {
                    // VkTransformMatrixKHR is row-major 3x4, Metal's matrix is 4 columns of 3
                    d.transformationMatrix.columns[ c ].elements[ r ] = src.transform.matrix[ r ][ c ];
                }
            }
            // the low VkGeometryInstanceFlagBitsKHR have the values of MTLAccelerationStructureInstanceOptions
            d.options = MTLAccelerationStructureInstanceOptions( src.flags & 0xF );
            d.userID  = src.instanceCustomIndex;

            auto byAddr = s.asByAddress.find( src.accelerationStructureReference );
            if( byAddr != s.asByAddress.end() )
            {
                auto bm = s.as.find( byAddr->second );
                if( bm != s.as.end() && bm->second.as )
                {
                    auto [ bi, added ] = blasIndex.try_emplace( byAddr->second, uint32_t( blasList.count ) );
                    if( added )
                    {
                        [blasList addObject:bm->second.as];
                    }
                    d.accelerationStructureIndex = bi->second;
                    d.mask                       = src.mask;
                    continue;
                }
            }
            // keep the instance index (the shaders index per-instance data by it), but hide it
            LogOnce( "a top level instance references an unknown bottom level structure" );
        }

        if( blasList.count == 0 )
        {
            LogOnce( "no placeholder bottom level structure, a top level build was skipped" );
            continue;
        }

        MTLInstanceAccelerationStructureDescriptor* d = [MTLInstanceAccelerationStructureDescriptor descriptor];
        d.instanceDescriptorType          = MTLAccelerationStructureInstanceDescriptorTypeUserID;
        d.instanceDescriptorBuffer        = instBuf;
        d.instanceDescriptorBufferOffset  = 0;
        d.instanceDescriptorStride        = sizeof( MTLAccelerationStructureUserIDInstanceDescriptor );
        d.instanceCount                   = count;
        d.instancedAccelerationStructures = blasList;
        d.usage = t.fastTrace ? MTLAccelerationStructureUsageNone : MTLAccelerationStructureUsagePreferFastBuild;

        const MTLAccelerationStructureSizes sizes = [s.mtlDevice accelerationStructureSizesWithDescriptor:d];
        EnsureAS( m, sizes.accelerationStructureSize );
        if( !m.as )
        {
            continue;
        }
        EnsureScratch( scratchOffset + Align256( std::max< NSUInteger >( sizes.buildScratchBufferSize, 1 ) ) );

        [ae buildAccelerationStructure:m.as
                            descriptor:d
                         scratchBuffer:s.scratch
                   scratchBufferOffset:scratchOffset];
        scratchOffset += Align256( std::max< NSUInteger >( sizes.buildScratchBufferSize, 1 ) );
    }
    [ae updateFence:s.fence];
    [ae endEncoding];
    s.pendingTlas.clear();
    return true;
}

}

// ================================================================================================
// Vulkan call interception

namespace
{
// translated graphics shaders are cached by module and pipeline layout handle; Vulkan may
// reuse a handle value after the object is destroyed
void ForgetGraphicsLibraries( uint64_t handle );
}

VkResult RTGL1::Metal::CreateDescriptorSetLayout( VkDevice                               device,
                                                  const VkDescriptorSetLayoutCreateInfo* ci,
                                                  const VkAllocationCallbacks*           alloc,
                                                  VkDescriptorSetLayout*                 out )
{
    const VkResult r = vkCreateDescriptorSetLayout( device, ci, alloc, out );
    if( r == VK_SUCCESS )
    {
        auto li = std::make_shared< LayoutInfo >();
        for( uint32_t i = 0; i < ci->bindingCount; i++ )
        {
            const auto& b = ci->pBindings[ i ];
            li->bindings.push_back( { b.binding, b.descriptorType, b.descriptorCount } );
            li->hasAccelerationStructure |= b.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        }
        li->slots = MetalGen::AssignSlots( li->bindings );

        std::lock_guard g( S().lock );
        S().layouts[ *out ] = std::move( li );
    }
    return r;
}

void RTGL1::Metal::DestroyDescriptorSetLayout( VkDevice device, VkDescriptorSetLayout l, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().layouts.erase( l );
    }
    vkDestroyDescriptorSetLayout( device, l, alloc );
}

VkResult RTGL1::Metal::CreatePipelineLayout( VkDevice                          device,
                                             const VkPipelineLayoutCreateInfo* ci,
                                             const VkAllocationCallbacks*      alloc,
                                             VkPipelineLayout*                 out )
{
    const VkResult r = vkCreatePipelineLayout( device, ci, alloc, out );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        auto&           v = S().pipelineLayouts[ *out ];
        v.clear();
        for( uint32_t i = 0; i < ci->setLayoutCount; i++ )
        {
            auto it = S().layouts.find( ci->pSetLayouts[ i ] );
            v.push_back( it != S().layouts.end() ? it->second : nullptr );
        }
    }
    return r;
}

void RTGL1::Metal::DestroyPipelineLayout( VkDevice device, VkPipelineLayout l, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().pipelineLayouts.erase( l );
        ForgetGraphicsLibraries( uint64_t( uintptr_t( l ) ) );
    }
    vkDestroyPipelineLayout( device, l, alloc );
}

VkResult RTGL1::Metal::AllocateDescriptorSets( VkDevice device, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* out )
{
    const VkResult r = vkAllocateDescriptorSets( device, ai, out );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        for( uint32_t i = 0; i < ai->descriptorSetCount; i++ )
        {
            SetInfo si = {};
            auto    it = S().layouts.find( ai->pSetLayouts[ i ] );
            si.layout  = it != S().layouts.end() ? it->second : nullptr;
            si.pool    = ai->descriptorPool;
            S().sets[ out[ i ] ] = std::move( si );
        }
    }
    return r;
}

VkResult RTGL1::Metal::FreeDescriptorSets( VkDevice device, VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets )
{
    {
        std::lock_guard g( S().lock );
        for( uint32_t i = 0; i < count; i++ )
        {
            S().sets.erase( sets[ i ] );
        }
    }
    return vkFreeDescriptorSets( device, pool, count, sets );
}

static void ForgetPoolSets( VkDescriptorPool pool )
{
    std::lock_guard g( S().lock );
    for( auto it = S().sets.begin(); it != S().sets.end(); )
    {
        it = it->second.pool == pool ? S().sets.erase( it ) : std::next( it );
    }
}

void RTGL1::Metal::DestroyDescriptorPool( VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* alloc )
{
    ForgetPoolSets( pool );
    vkDestroyDescriptorPool( device, pool, alloc );
}

VkResult RTGL1::Metal::ResetDescriptorPool( VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags )
{
    ForgetPoolSets( pool );
    return vkResetDescriptorPool( device, pool, flags );
}

void RTGL1::Metal::UpdateDescriptorSets( VkDevice                    device,
                                         uint32_t                    writeCount,
                                         const VkWriteDescriptorSet* writes,
                                         uint32_t                    copyCount,
                                         const VkCopyDescriptorSet*  copies )
{
    vkUpdateDescriptorSets( device, writeCount, writes, copyCount, copies );

    std::lock_guard g( S().lock );
    if( copyCount > 0 )
    {
        LogOnce( "descriptor set copies aren't tracked" );
    }

    for( uint32_t w = 0; w < writeCount; w++ )
    {
        const VkWriteDescriptorSet& wr = writes[ w ];
        auto                        it = S().sets.find( wr.dstSet );
        if( it == S().sets.end() || !it->second.layout )
        {
            continue;
        }
        SetInfo& si = it->second;

        const MetalGen::SlotInfo* slot = si.layout->slots.Find( wr.dstBinding );
        if( !slot )
        {
            continue;
        }
        auto& vec = si.descs[ wr.dstBinding ];
        if( vec.size() < slot->count )
        {
            vec.resize( slot->count );
        }

        const VkWriteDescriptorSetAccelerationStructureKHR* asWrite = nullptr;
        for( auto* n = static_cast< const VkBaseInStructure* >( wr.pNext ); n; n = n->pNext )
        {
            if( n->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR )
            {
                asWrite = reinterpret_cast< const VkWriteDescriptorSetAccelerationStructureKHR* >( n );
            }
        }

        for( uint32_t k = 0; k < wr.descriptorCount; k++ )
        {
            const uint32_t e = wr.dstArrayElement + k;
            if( e >= vec.size() )
            {
                LogOnce( "a descriptor write past the end of its binding isn't tracked" );
                break;
            }
            Descriptor& d = vec[ e ];
            d             = {};
            switch( wr.descriptorType )
            {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    if( wr.pImageInfo )
                    {
                        d.view    = wr.pImageInfo[ k ].imageView;
                        d.sampler = wr.pImageInfo[ k ].sampler;
                    }
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    if( wr.pBufferInfo )
                    {
                        d.buffer = wr.pBufferInfo[ k ].buffer;
                        d.offset = wr.pBufferInfo[ k ].offset;
                    }
                    break;
                case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                    if( asWrite && k < asWrite->accelerationStructureCount )
                    {
                        d.as = asWrite->pAccelerationStructures[ k ];
                    }
                    break;
                default: break;
            }
        }
        si.version++;
    }
}

VkResult RTGL1::Metal::CreateSampler( VkDevice device, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks* alloc, VkSampler* out )
{
    const VkResult r = vkCreateSampler( device, ci, alloc, out );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        SamplerInfo     si = {};
        si.ci              = *ci;
        si.ci.pNext        = nullptr;
        S().samplers[ *out ] = std::move( si );
    }
    return r;
}

void RTGL1::Metal::DestroySampler( VkDevice device, VkSampler sampler, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().samplers.erase( sampler );
    }
    vkDestroySampler( device, sampler, alloc );
}

VkResult RTGL1::Metal::CreateBuffer( VkDevice device, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* alloc, VkBuffer* out )
{
    const VkResult r = vkCreateBuffer( device, ci, alloc, out );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        S().buffers[ *out ].size = ci->size;
    }
    return r;
}

void RTGL1::Metal::DestroyBuffer( VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        auto            it = S().buffers.find( buffer );
        if( it != S().buffers.end() )
        {
            if( it->second.address )
            {
                auto a = S().buffersByAddress.find( it->second.address );
                if( a != S().buffersByAddress.end() && a->second == buffer )
                {
                    S().buffersByAddress.erase( a );
                }
            }
            S().buffers.erase( it );
        }
    }
    vkDestroyBuffer( device, buffer, alloc );
}

VkResult RTGL1::Metal::BindBufferMemory( VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset )
{
    const VkResult r = vkBindBufferMemory( device, buffer, memory, offset );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        auto&           bi = S().buffers[ buffer ];
        bi.memory          = memory;
        bi.memoryOffset    = offset;
    }
    return r;
}

VkDeviceAddress RTGL1::Metal::GetBufferDeviceAddress( VkDevice device, const VkBufferDeviceAddressInfo* info )
{
    const VkDeviceAddress a = vkGetBufferDeviceAddress( device, info );
    if( a != 0 )
    {
        std::lock_guard g( S().lock );
        S().buffers[ info->buffer ].address = a;
        S().buffersByAddress[ a ]           = info->buffer;
    }
    return a;
}

void RTGL1::Metal::FreeMemory( VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        auto            it = S().memoryBuffers.find( memory );
        if( it != S().memoryBuffers.end() )
        {
            RemoveResident( it->second );
            S().memoryBuffers.erase( it );
        }
    }
    vkFreeMemory( device, memory, alloc );
}

void RTGL1::Metal::DestroyImageView( VkDevice device, VkImageView view, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        auto            it = S().viewTextures.find( view );
        if( it != S().viewTextures.end() )
        {
            RemoveResident( it->second );
            if( it->second.parentTexture )
            {
                RemoveResident( it->second.parentTexture );
            }
            S().viewTextures.erase( it );
        }
    }
    vkDestroyImageView( device, view, alloc );
}

void RTGL1::Metal::ForgetImage( VkImage image )
{
    std::lock_guard g( S().lock );
    auto            it = S().imageTextures.find( image );
    if( it != S().imageTextures.end() )
    {
        RemoveResident( it->second );
        S().imageTextures.erase( it );
    }
}

void RTGL1::Metal::DestroyImage( VkDevice device, VkImage image, const VkAllocationCallbacks* alloc )
{
    ForgetImage( image );
    vkDestroyImage( device, image, alloc );
}

VkResult RTGL1::Metal::CreateShaderModule( VkDevice                        device,
                                           const VkShaderModuleCreateInfo* ci,
                                           const VkAllocationCallbacks*    alloc,
                                           VkShaderModule*                 out )
{
    const VkResult r = vkCreateShaderModule( device, ci, alloc, out );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        S().modules[ *out ].assign( ci->pCode, ci->pCode + ci->codeSize / sizeof( uint32_t ) );
    }
    return r;
}

void RTGL1::Metal::DestroyShaderModule( VkDevice device, VkShaderModule module, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().modules.erase( module );
        ForgetGraphicsLibraries( uint64_t( uintptr_t( module ) ) );
    }
    vkDestroyShaderModule( device, module, alloc );
}

VkResult RTGL1::Metal::CreateComputePipelines( VkDevice                           device,
                                               VkPipelineCache                    cache,
                                               uint32_t                           count,
                                               const VkComputePipelineCreateInfo* cis,
                                               const VkAllocationCallbacks*       alloc,
                                               VkPipeline*                        out )
{
    const VkResult r = vkCreateComputePipelines( device, cache, count, cis, alloc, out );
    if( r == VK_SUCCESS )
    {
        std::lock_guard g( S().lock );
        for( uint32_t i = 0; i < count; i++ )
        {
            PipelineInfo pi = {};
            pi.module       = cis[ i ].stage.module;
            pi.entry        = cis[ i ].stage.pName ? cis[ i ].stage.pName : "main";
            pi.layout       = cis[ i ].layout;
            if( const VkSpecializationInfo* sp = cis[ i ].stage.pSpecializationInfo )
            {
                pi.specEntries.assign( sp->pMapEntries, sp->pMapEntries + sp->mapEntryCount );
                const auto* data = static_cast< const uint8_t* >( sp->pData );
                pi.specData.assign( data, data + sp->dataSize );
            }
            S().pipelines[ out[ i ] ] = std::move( pi );
        }
    }
    return r;
}

void RTGL1::Metal::DestroyPipeline( VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().pipelines.erase( pipeline );
        S().gpipelines.erase( pipeline );
    }
    vkDestroyPipeline( device, pipeline, alloc );
}

VkResult RTGL1::Metal::CreateRenderPass( VkDevice                      device,
                                         const VkRenderPassCreateInfo* ci,
                                         const VkAllocationCallbacks*  alloc,
                                         VkRenderPass*                 out )
{
    const VkResult r = vkCreateRenderPass( device, ci, alloc, out );
    if( r != VK_SUCCESS )
    {
        return r;
    }

    RenderPassInfo info;
    for( uint32_t i = 0; i < ci->attachmentCount; i++ )
    {
        info.attachments.push_back(
            { ci->pAttachments[ i ].format, ci->pAttachments[ i ].loadOp, ci->pAttachments[ i ].storeOp } );
    }
    if( ci->subpassCount == 1 )
    {
        const VkSubpassDescription& sp = ci->pSubpasses[ 0 ];
        for( uint32_t i = 0; i < sp.colorAttachmentCount; i++ )
        {
            info.colorRefs.push_back( sp.pColorAttachments[ i ].attachment );
        }
        if( sp.pDepthStencilAttachment )
        {
            info.depthRef = sp.pDepthStencilAttachment->attachment;
        }
        if( sp.inputAttachmentCount > 0 || sp.pResolveAttachments )
        {
            info.supported = false;
        }
    }
    else
    {
        info.supported = false;
    }
    for( auto* n = static_cast< const VkBaseInStructure* >( ci->pNext ); n; n = n->pNext )
    {
        if( n->sType == VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO )
        {
            const auto* mv = reinterpret_cast< const VkRenderPassMultiviewCreateInfo* >( n );
            if( mv->subpassCount > 0 && mv->pViewMasks )
            {
                info.viewMask = mv->pViewMasks[ 0 ];
            }
        }
    }

    std::lock_guard g( S().lock );
    S().renderPasses[ *out ] = std::move( info );
    return r;
}

void RTGL1::Metal::DestroyRenderPass( VkDevice device, VkRenderPass rp, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().renderPasses.erase( rp );
    }
    vkDestroyRenderPass( device, rp, alloc );
}

VkResult RTGL1::Metal::CreateFramebuffer( VkDevice                       device,
                                          const VkFramebufferCreateInfo* ci,
                                          const VkAllocationCallbacks*   alloc,
                                          VkFramebuffer*                 out )
{
    const VkResult r = vkCreateFramebuffer( device, ci, alloc, out );
    if( r == VK_SUCCESS )
    {
        FramebufferInfo info;
        info.renderPass = ci->renderPass;
        info.views.assign( ci->pAttachments, ci->pAttachments + ci->attachmentCount );
        info.width  = ci->width;
        info.height = ci->height;
        info.layers = ci->layers;

        std::lock_guard g( S().lock );
        S().framebuffers[ *out ] = std::move( info );
    }
    return r;
}

void RTGL1::Metal::DestroyFramebuffer( VkDevice device, VkFramebuffer fb, const VkAllocationCallbacks* alloc )
{
    {
        std::lock_guard g( S().lock );
        S().framebuffers.erase( fb );
    }
    vkDestroyFramebuffer( device, fb, alloc );
}

VkResult RTGL1::Metal::CreateGraphicsPipelines( VkDevice                            device,
                                                VkPipelineCache                     cache,
                                                uint32_t                            count,
                                                const VkGraphicsPipelineCreateInfo* cis,
                                                const VkAllocationCallbacks*        alloc,
                                                VkPipeline*                         out )
{
    const VkResult r = vkCreateGraphicsPipelines( device, cache, count, cis, alloc, out );
    if( r != VK_SUCCESS )
    {
        return r;
    }

    std::lock_guard g( S().lock );
    for( uint32_t i = 0; i < count; i++ )
    {
        const VkGraphicsPipelineCreateInfo& ci = cis[ i ];
        GraphicsPipelineInfo                gp;

        for( uint32_t k = 0; k < ci.stageCount; k++ )
        {
            const VkPipelineShaderStageCreateInfo& st = ci.pStages[ k ];
            ShaderStageInfo                        si;
            si.module = st.module;
            si.entry  = st.pName ? st.pName : "main";
            if( st.pSpecializationInfo )
            {
                const VkSpecializationInfo& sp = *st.pSpecializationInfo;
                si.specEntries.assign( sp.pMapEntries, sp.pMapEntries + sp.mapEntryCount );
                const auto* data = static_cast< const uint8_t* >( sp.pData );
                si.specData.assign( data, data + sp.dataSize );
            }
            if( st.stage == VK_SHADER_STAGE_VERTEX_BIT )
            {
                gp.vs = std::move( si );
            }
            else if( st.stage == VK_SHADER_STAGE_FRAGMENT_BIT )
            {
                gp.fs    = std::move( si );
                gp.hasFs = true;
            }
            else
            {
                gp.failed = true; // geometry / tessellation: not supported
            }
        }

        if( const auto* vi = ci.pVertexInputState )
        {
            gp.bindings.assign( vi->pVertexBindingDescriptions,
                                vi->pVertexBindingDescriptions + vi->vertexBindingDescriptionCount );
            gp.attributes.assign( vi->pVertexAttributeDescriptions,
                                  vi->pVertexAttributeDescriptions + vi->vertexAttributeDescriptionCount );
        }
        if( ci.pInputAssemblyState )
        {
            gp.topology = ci.pInputAssemblyState->topology;
        }
        if( const auto* rs = ci.pRasterizationState )
        {
            gp.cullMode          = rs->cullMode;
            gp.frontFace         = rs->frontFace;
            gp.depthBias         = rs->depthBiasEnable;
            gp.depthBiasConstant = rs->depthBiasConstantFactor;
            gp.depthBiasSlope    = rs->depthBiasSlopeFactor;
            gp.depthBiasClamp    = rs->depthBiasClamp;
            gp.wireframe         = rs->polygonMode == VK_POLYGON_MODE_LINE;
        }
        if( const auto* ds = ci.pDepthStencilState )
        {
            gp.depthTest    = ds->depthTestEnable;
            gp.depthWrite   = ds->depthTestEnable && ds->depthWriteEnable;
            gp.depthCompare = ds->depthCompareOp;
        }
        if( const auto* cb = ci.pColorBlendState )
        {
            gp.blends.assign( cb->pAttachments, cb->pAttachments + cb->attachmentCount );
        }
        if( const auto* dy = ci.pDynamicState )
        {
            for( uint32_t k = 0; k < dy->dynamicStateCount; k++ )
            {
                gp.dynamicViewport |= dy->pDynamicStates[ k ] == VK_DYNAMIC_STATE_VIEWPORT;
                gp.dynamicScissor |= dy->pDynamicStates[ k ] == VK_DYNAMIC_STATE_SCISSOR;
            }
        }
        if( const auto* vp = ci.pViewportState )
        {
            if( vp->pViewports && vp->viewportCount > 0 )
            {
                gp.viewport = vp->pViewports[ 0 ];
            }
            if( vp->pScissors && vp->scissorCount > 0 )
            {
                gp.scissor = vp->pScissors[ 0 ];
            }
        }
        gp.layout     = ci.layout;
        gp.renderPass = ci.renderPass;
        auto rp       = S().renderPasses.find( ci.renderPass );
        if( rp != S().renderPasses.end() && rp->second.viewMask != 0 )
        {
            gp.viewCount = uint32_t( __builtin_popcount( rp->second.viewMask ) );
        }

        S().gpipelines[ out[ i ] ] = std::move( gp );
    }
    return r;
}

// ================================================================================================
// native path

namespace
{
void CreateHelpers();
void ShutdownGraphics();
}

bool RTGL1::Metal::IsEnabled()
{
    return S().enabled;
}

bool RTGL1::Metal::IsInSection()
{
    return S().cb != nil;
}

void RTGL1::Metal::Disable( const char* reason )
{
    std::lock_guard g( S().lock );
    if( S().enabled )
    {
        Log( "native path tracing turned off: %s", reason );
    }
    S().enabled = false;
}

bool RTGL1::Metal::Init( VkDevice device, std::shared_ptr< CommandBufferManager > cmdManager )
{
    const char* e = getenv( "RTGL1_METAL_RT" );
    if( !( e && e[ 0 ] == '1' ) )
    {
        return false;
    }

    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();

        s.device     = device;
        s.cmdManager = std::move( cmdManager );
        s.exportFn   = reinterpret_cast< PFN_vkExportMetalObjectsEXT >(
            vkGetDeviceProcAddr( device, "vkExportMetalObjectsEXT" ) );
        if( !s.exportFn )
        {
            Log( "vkExportMetalObjectsEXT is not available, native path tracing is off" );
            return false;
        }

        VkExportMetalDeviceInfoEXT  dInfo = { VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT };
        VkExportMetalObjectsInfoEXT info  = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
        info.pNext                        = &dInfo;
        s.exportFn( device, &info );
        s.mtlDevice = dInfo.mtlDevice;

        if( !s.mtlDevice || !s.mtlDevice.supportsRaytracing )
        {
            Log( "the Metal device doesn't support ray tracing, native path tracing is off" );
            return false;
        }
        if( @available( macOS 15.0, * ) )
        {
        }
        else
        {
            Log( "native path tracing needs macOS 15 or newer" );
            return false;
        }

        s.queue       = [s.mtlDevice newCommandQueue];
        s.queue.label = @"RTGL1 native";
        s.event       = [s.mtlDevice newSharedEvent];
        s.fence       = [s.mtlDevice newFence];
        s.blasFence   = [s.mtlDevice newFence];
        s.passFence   = [s.mtlDevice newFence];
        if( !s.queue || !s.event || !s.fence || !s.blasFence || !s.passFence )
        {
            Log( "couldn't create a Metal queue / event / fence" );
            return false;
        }
        s.event.signaledValue = 0;
        s.timeline            = 0;

        // a Vulkan timeline semaphore on the same event, for the ordering with the Vulkan work
        VkImportMetalSharedEventInfoEXT imp = { VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT };
        imp.mtlSharedEvent                  = s.event;
        VkSemaphoreTypeCreateInfo type      = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        type.pNext                          = &imp;
        type.semaphoreType                  = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci           = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        sci.pNext                           = &type;
        if( vkCreateSemaphore( device, &sci, nullptr, &s.semaphore ) != VK_SUCCESS )
        {
            s.semaphore = VK_NULL_HANDLE;
            Log( "couldn't import the shared event into Vulkan" );
            return false;
        }

        if( @available( macOS 15.0, * ) )
        {
            MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
            rd.label                      = @"RTGL1 native";
            rd.initialCapacity            = 4096;
            NSError* err                  = nil;
            id< MTLResidencySet > rs      = [s.mtlDevice newResidencySetWithDescriptor:rd error:&err];
            if( !rs )
            {
                Log( "couldn't create a residency set: %s", err ? err.localizedDescription.UTF8String : "?" );
                return false;
            }
            [s.queue addResidencySet:rs];
            s.residency = rs;
        }

        {
            const char* p = getenv( "RTGL1_PROFILE" );
            s.profile     = p && p[ 0 ] == '1';
            const char* c = getenv( "RTGL1_METAL_COMPUTE" );
            s.computeEnv  = !( c && c[ 0 ] == '0' );
            const char* f = getenv( "RTGL1_METAL_FASTMATH" );
            s.fastMath    = f && f[ 0 ] == '1';
            const char* ra = getenv( "RTGL1_METAL_RASTER" );
            s.rasterEnv    = !( ra && ra[ 0 ] == '0' );
            const char* r = getenv( "RTGL1_METAL_AS_REBUILD_EVERY" );
            if( r && r[ 0 ] )
            {
                s.asRebuildEvery = uint32_t( std::clamp( atoi( r ), 1, 1000 ) );
            }
        }

        CreateHelpers();

        s.enabled = true;
        Log( "native path tracing enabled (RTGL1_METAL_RT=1) on %s%s", s.mtlDevice.name.UTF8String,
             s.computeEnv ? ", with the denoiser and other compute passes" : "" );
        return true;
    }
}

void RTGL1::Metal::Shutdown()
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( s.lastCb )
        {
            [s.lastCb waitUntilCompleted];
            s.lastCb = nil;
        }
        if( s.semaphore != VK_NULL_HANDLE )
        {
            vkDestroySemaphore( s.device, s.semaphore, nullptr );
            s.semaphore = VK_NULL_HANDLE;
        }
        for( auto& [ set, si ] : s.sets )
        {
            si.argBuffers.clear();
            si.argBufferCur   = -1;
            si.encodedVersion = 0;
        }
        s.as.clear();
        s.asByAddress.clear();
        s.pendingBlas.clear();
        s.pendingTlas.clear();
        s.viewTextures.clear();
        s.imageTextures.clear();
        s.memoryBuffers.clear();
        for( auto& [ smp, si ] : s.samplers )
        {
            si.mtl = nil;
        }
        for( auto& [ p, pi ] : s.pipelines )
        {
            pi.mtl = nil;
        }
        s.resident.clear();
        s.residency  = nil;
        s.scratch    = nil;
        s.fence      = nil;
        s.blasFence  = nil;
        s.passFence  = nil;
        s.profWaitCb = nil;
        s.profCbs.clear();
        ShutdownGraphics();
        s.dummyBlas  = nil;
        s.event      = nil;
        s.queue      = nil;
        s.mtlDevice  = nil;
        s.cmdManager = nullptr;
        s.enabled    = false;
    }
}

namespace
{

struct PrepareJob
{
    VkPipeline                              pipeline = VK_NULL_HANDLE;
    std::string                             name;
    std::vector< uint32_t >                 spirv;
    std::vector< MetalGen::SetSlots >       sets;
    std::vector< VkSpecializationMapEntry > specEntries;
    std::vector< uint8_t >                  specData;
    bool                                    fastMath = false;
};

struct PrepareResult
{
    id< MTLComputePipelineState > ps;
    MTLSize                       threadsPerGroup = { 1, 1, 1 };
    std::string                   error;
};

// under the lock
bool MakePrepareJob( VkPipeline pipeline, PrepareJob& job, std::string& error )
{
    State& s   = S();
    auto   pit = s.pipelines.find( pipeline );
    if( pit == s.pipelines.end() )
    {
        error = "unknown pipeline";
        return false;
    }
    const PipelineInfo& pi = pit->second;

    auto mit = s.modules.find( pi.module );
    auto lit = s.pipelineLayouts.find( pi.layout );
    if( mit == s.modules.end() || lit == s.pipelineLayouts.end() )
    {
        error = "shader module or pipeline layout unknown";
        return false;
    }

    job.pipeline = pipeline;
    job.name     = pi.name;
    job.spirv    = mit->second;
    job.sets.clear();
    for( const auto& l : lit->second )
    {
        job.sets.push_back( l ? l->slots : MetalGen::SetSlots{} );
    }
    job.specEntries = pi.specEntries;
    job.specData    = pi.specData;
    job.fastMath    = s.fastMath;
    return true;
}

// Compiles translated MSL (retrying without the optional ray query hint added by MetalGen).
// Without the lock: can run on several threads at once.
id< MTLLibrary > CompileLibrary( id< MTLDevice >    device,
                                 const std::string& msl,
                                 bool               fastMath,
                                 const std::string& name,
                                 std::string&       error )
{
    NSError*           err = nil;
    MTLCompileOptions* co  = [MTLCompileOptions new];
    co.languageVersion     = MTLLanguageVersion3_1;
    if( @available( macOS 15.0, * ) )
    {
        // fast math lets the compiler assume no Inf / NaN, so the shaders' isnan / isinf
        // checks (that keep bad values out of the reservoirs and the denoiser history)
        // could be removed; relaxed keeps the other optimizations but honors Inf / NaN
        co.mathMode = fastMath ? MTLMathModeFast : MTLMathModeRelaxed;
    }
    id< MTLLibrary > lib = [device newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()]
                                                options:co
                                                  error:&err];
    if( !lib )
    {
        const std::string hint  = "    ip.assume_geometry_type(geometry_type::triangle);\n";
        std::string       plain = msl;
        const size_t      at    = plain.find( hint );
        if( at != std::string::npos )
        {
            plain.erase( at, hint.size() );
            NSError* err2 = nil;
            lib = [device newLibraryWithSource:[NSString stringWithUTF8String:plain.c_str()]
                                       options:co
                                         error:&err2];
            if( lib )
            {
                Log( "%s: compiled without the triangle-only ray query hint", name.c_str() );
            }
        }
    }
    if( !lib )
    {
        NSString* path = [NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:@"rtgl1-%s.metal", name.c_str()]];
        [[NSString stringWithUTF8String:msl.c_str()] writeToFile:path
                                                      atomically:YES
                                                        encoding:NSUTF8StringEncoding
                                                           error:nil];
        std::string msg = err ? err.localizedDescription.UTF8String : "?";
        error = "Metal compile failed (source saved to " + std::string( path.UTF8String ) +
                "): " + msg.substr( 0, 3000 );
    }
    return lib;
}

// The entry point with the specialization constants as function constants, by the types the
// shader declares (bool, int, uint, float; SPIRV-Cross keeps the constant_id as the index).
id< MTLFunction > MakeFunction( id< MTLLibrary >                               lib,
                                const std::string&                             entryPoint,
                                const std::vector< VkSpecializationMapEntry >& specEntries,
                                const std::vector< uint8_t >&                  specData,
                                std::string&                                   error )
{
    NSString*                  entry = [NSString stringWithUTF8String:entryPoint.c_str()];
    MTLFunctionConstantValues* fc    = [MTLFunctionConstantValues new];
    id< MTLFunction >          probe = [lib newFunctionWithName:entry];
    if( !probe )
    {
        error = "entry point '" + entryPoint + "' not found";
        return nil;
    }
    for( MTLFunctionConstant* c in probe.functionConstantsDictionary.allValues )
    {
        for( const VkSpecializationMapEntry& se : specEntries )
        {
            if( se.constantID != c.index || se.offset + se.size > specData.size() )
            {
                continue;
            }
            const uint8_t* src = specData.data() + se.offset;
            switch( c.type )
            {
                case MTLDataTypeBool: {
                    uint32_t raw = 0;
                    memcpy( &raw, src, std::min< size_t >( se.size, sizeof( raw ) ) );
                    const bool b = raw != 0;
                    [fc setConstantValue:&b type:MTLDataTypeBool atIndex:c.index];
                    break;
                }
                case MTLDataTypeInt:
                case MTLDataTypeUInt:
                case MTLDataTypeFloat:
                    if( se.size == 4 )
                    {
                        [fc setConstantValue:src type:c.type atIndex:c.index];
                    }
                    break;
                default: break;
            }
        }
    }

    NSError*          err = nil;
    id< MTLFunction > fn  = [lib newFunctionWithName:entry constantValues:fc error:&err];
    if( !fn )
    {
        error = "function '" + entryPoint + "' failed: " + ( err ? err.localizedDescription.UTF8String : "?" );
    }
    return fn;
}

// without the lock: can run on several threads at once
PrepareResult CompilePrepareJob( const PrepareJob& job, id< MTLDevice > device )
{
    @autoreleasepool
    {
        PrepareResult r;

        const MetalGen::ComputeShader sh =
            MetalGen::TranslateCompute( job.spirv.data(), job.spirv.size(), job.sets, 30100 );
        if( !sh.ok )
        {
            r.error = "SPIR-V to Metal translation failed: " + sh.error;
            return r;
        }

        id< MTLLibrary > lib = CompileLibrary( device, sh.msl, job.fastMath, job.name, r.error );
        if( !lib )
        {
            return r;
        }
        id< MTLFunction > fn = MakeFunction( lib, sh.entryPoint, job.specEntries, job.specData, r.error );
        if( !fn )
        {
            return r;
        }

        NSError*                      err = nil;
        MTLComputePipelineDescriptor* pd  = [MTLComputePipelineDescriptor new];
        pd.computeFunction                = fn;
        pd.label                          = [NSString stringWithUTF8String:job.name.c_str()];
        id< MTLComputePipelineState > ps  = [device newComputePipelineStateWithDescriptor:pd
                                                                                 options:MTLPipelineOptionNone
                                                                              reflection:nil
                                                                                   error:&err];
        const NSUInteger threads = NSUInteger( sh.localSize[ 0 ] ) * sh.localSize[ 1 ] * sh.localSize[ 2 ];
        if( !ps || ps.maxTotalThreadsPerThreadgroup < threads )
        {
            r.error = std::string( "pipeline failed: " ) +
                      ( ps ? "too many threads per group" : ( err ? err.localizedDescription.UTF8String : "?" ) );
            return r;
        }

        r.ps              = ps;
        r.threadsPerGroup = MTLSizeMake( sh.localSize[ 0 ], sh.localSize[ 1 ], sh.localSize[ 2 ] );
        return r;
    }
}

// under the lock
bool ApplyPrepareResult( const PrepareJob& job, const PrepareResult& r, bool logSuccess )
{
    auto pit = S().pipelines.find( job.pipeline );
    if( pit == S().pipelines.end() )
    {
        return false; // destroyed meanwhile
    }
    PipelineInfo& pi = pit->second;
    if( !r.ps )
    {
        pi.failed = true;
        Log( "%s: %s", job.name.c_str(), r.error.c_str() );
        return false;
    }
    pi.mtl             = r.ps;
    pi.threadsPerGroup = r.threadsPerGroup;
    pi.failed          = false;
    if( logSuccess )
    {
        Log( "%s: ready", job.name.c_str() );
    }
    return true;
}

std::string DefaultPipelineName( VkPipeline pipeline )
{
    char buf[ 48 ];
    snprintf( buf, sizeof( buf ), "compute pipeline %llx", ( unsigned long long )( uintptr_t )pipeline );
    return buf;
}

// under the lock
bool PreparePipelineLocked( VkPipeline pipeline, bool logSuccess )
{
    State& s = S();
    if( !s.enabled )
    {
        return false;
    }
    PrepareJob  job;
    std::string error;
    if( !MakePrepareJob( pipeline, job, error ) )
    {
        Log( "pipeline %p: %s", ( void* )pipeline, error.c_str() );
        return false;
    }
    return ApplyPrepareResult( job, CompilePrepareJob( job, s.mtlDevice ), logSuccess );
}

}

bool RTGL1::Metal::PreparePipeline( VkPipeline pipeline, const char* debugName )
{
    std::lock_guard g( S().lock );
    auto            pit = S().pipelines.find( pipeline );
    if( pit != S().pipelines.end() )
    {
        pit->second.name = debugName ? debugName : DefaultPipelineName( pipeline );
    }
    return PreparePipelineLocked( pipeline, true );
}

void RTGL1::Metal::PrepareAllPipelines()
{
    std::vector< PrepareJob > jobs;
    id< MTLDevice >           device;
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( !s.enabled || !s.computeEnv )
        {
            return;
        }
        device = s.mtlDevice;
        for( auto& [ p, pi ] : s.pipelines )
        {
            if( pi.mtl || pi.failed )
            {
                continue;
            }
            if( pi.name.empty() )
            {
                pi.name = DefaultPipelineName( p );
            }
            PrepareJob  job;
            std::string error;
            if( MakePrepareJob( p, job, error ) )
            {
                jobs.push_back( std::move( job ) );
            }
        }
    }
    if( jobs.empty() )
    {
        return;
    }

    std::vector< PrepareResult > results( jobs.size() );
    std::atomic< size_t >        next{ 0 };
    const unsigned               threadCount =
        std::clamp( std::thread::hardware_concurrency(), 1u, unsigned( std::min< size_t >( jobs.size(), 8 ) ) );
    std::vector< std::thread > threads;
    for( unsigned t = 0; t < threadCount; t++ )
    {
        threads.emplace_back( [ & ] {
            for( size_t i = next++; i < jobs.size(); i = next++ )
            {
                results[ i ] = CompilePrepareJob( jobs[ i ], device );
            }
        } );
    }
    for( auto& t : threads )
    {
        t.join();
    }

    std::lock_guard g( S().lock );
    size_t          ok = 0;
    for( size_t i = 0; i < jobs.size(); i++ )
    {
        ok += ApplyPrepareResult( jobs[ i ], results[ i ], false ) ? 1 : 0;
    }
    Log( "%zu compute pipelines prepared for Metal, %zu failed", ok, jobs.size() - ok );
}

void RTGL1::Metal::RegisterAccelerationStructureAddress( VkAccelerationStructureKHR as, VkDeviceAddress address )
{
    std::lock_guard g( S().lock );
    S().asByAddress[ address ] = as;
}

void RTGL1::Metal::ForgetAccelerationStructure( VkAccelerationStructureKHR as )
{
    std::lock_guard g( S().lock );
    State&          s  = S();
    auto            it = s.as.find( as );
    if( it != s.as.end() )
    {
        if( it->second.as )
        {
            RemoveResident( ( id< MTLResource > )it->second.as );
        }
        s.as.erase( it );
        s.asGeneration++;
    }
    for( auto a = s.asByAddress.begin(); a != s.asByAddress.end(); )
    {
        a = a->second == as ? s.asByAddress.erase( a ) : std::next( a );
    }
    // builds of it that are still queued
    s.pendingBlas.erase( std::remove_if( s.pendingBlas.begin(), s.pendingBlas.end(),
                                         [ as ]( const BlasBuild& b ) { return b.dst == as; } ),
                         s.pendingBlas.end() );
    s.pendingTlas.erase( std::remove_if( s.pendingTlas.begin(), s.pendingTlas.end(),
                                         [ as ]( const TlasBuild& t ) { return t.dst == as; } ),
                         s.pendingTlas.end() );
}

void RTGL1::Metal::QueueBottomLevelBuilds( uint32_t                                                count,
                                           const VkAccelerationStructureBuildGeometryInfoKHR*      infos,
                                           const VkAccelerationStructureBuildRangeInfoKHR* const* ranges )
{
    std::lock_guard g( S().lock );
    for( uint32_t i = 0; i < count; i++ )
    {
        const auto& info = infos[ i ];
        BlasBuild   b;
        b.dst    = info.dstAccelerationStructure;
        b.flags  = info.flags;
        b.update = info.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        for( uint32_t k = 0; k < info.geometryCount; k++ )
        {
            b.geoms.push_back( info.pGeometries ? info.pGeometries[ k ] : *info.ppGeometries[ k ] );
            b.geoms.back().pNext = nullptr;
            b.ranges.push_back( ranges[ i ][ k ] );
        }
        S().pendingBlas.push_back( std::move( b ) );
    }
}

void RTGL1::Metal::QueueTopLevelBuild( VkAccelerationStructureKHR                dst,
                                       const VkAccelerationStructureInstanceKHR* instances,
                                       uint32_t                                  instanceCount,
                                       bool                                      preferFastTrace )
{
    std::lock_guard g( S().lock );
    TlasBuild       t;
    t.dst       = dst;
    t.fastTrace = preferFastTrace;
    t.instances.assign( instances, instances + instanceCount );
    S().pendingTlas.push_back( std::move( t ) );
}

namespace
{

const char* Intern( const char* name )
{
    std::string full = std::string( "metal: " ) + ( name ? name : "?" );
    return S().internedNames.insert( std::move( full ) ).first->c_str();
}

void CommitResidency()
{
    State& s = S();
    if( s.residencyDirty )
    {
        if( @available( macOS 15.0, * ) )
        {
            [( id< MTLResidencySet > )s.residency commit];
        }
        s.residencyDirty = false;
    }
}

// push constant data is kept in 256-byte arrays; Metal's structs may be padded to 16 bytes
NSUInteger PushLength( uint32_t size )
{
    return std::min< NSUInteger >( ( NSUInteger( size ) + 15 ) & ~NSUInteger( 15 ), 256 );
}

// ---- encoders of a section
//
// One encoder at a time; each waits for passFence at its start and updates it at its end,
// so all the work of a section runs in the order it was recorded (resources reached
// through argument buffers aren't tracked by Metal). Within a compute encoder, a memory
// barrier follows each dispatch.

void WaitSectionFences( id< MTLComputeCommandEncoder > e )
{
    State& s = S();
    [e waitForFence:s.passFence];
    if( s.waitASFence )
    {
        [e waitForFence:s.fence];
    }
}

// ends the compute or blit encoder (a render encoder is ended by EndRenderPass)
void EndNonRenderEncoder()
{
    State& s = S();
    if( s.encoder )
    {
        [s.encoder updateFence:s.passFence];
        [s.encoder endEncoding];
        s.encoder = nil;
    }
    if( s.blitEnc )
    {
        [s.blitEnc updateFence:s.passFence];
        [s.blitEnc endEncoding];
        s.blitEnc = nil;
    }
}

id< MTLComputeCommandEncoder > ComputeEnc()
{
    State& s = S();
    assert( !s.renderEnc );
    if( s.blitEnc )
    {
        EndNonRenderEncoder();
    }
    if( !s.encoder )
    {
        s.encoder       = [s.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        s.encoder.label = @"RTGL1 native compute";
        WaitSectionFences( s.encoder );
    }
    return s.encoder;
}

id< MTLBlitCommandEncoder > BlitEnc()
{
    State& s = S();
    assert( !s.renderEnc );
    if( s.encoder )
    {
        EndNonRenderEncoder();
    }
    if( !s.blitEnc )
    {
        s.blitEnc       = [s.cb blitCommandEncoder];
        s.blitEnc.label = @"RTGL1 native copies";
        [s.blitEnc waitForFence:s.passFence];
        if( s.waitASFence )
        {
            [s.blitEnc waitForFence:s.fence];
        }
    }
    return s.blitEnc;
}

// profiling: ends the current command buffer, so its GPU time is the work since the last mark
void SplitForProfiling( const char* internedName )
{
    State& s = S();
    if( s.renderEnc )
    {
        return; // no marks inside a render pass
    }
    EndNonRenderEncoder();
    CommitResidency();
    [s.cb commit];
    s.profCbs.push_back( { s.cb, internedName } );

    s.cb       = [s.queue commandBuffer];
    s.cb.label = @"RTGL1 native (profiled)";
}

void EncodeDispatch( PipelineInfo&          pi,
                     const VkDescriptorSet* sets,
                     uint32_t               setCount,
                     const void*            push,
                     uint32_t               pushSize,
                     uint32_t               groupsX,
                     uint32_t               groupsY,
                     uint32_t               groupsZ )
{
    id< MTLComputeCommandEncoder > e = ComputeEnc();
    [e setComputePipelineState:pi.mtl];
    for( uint32_t i = 0; i < setCount; i++ )
    {
        if( sets[ i ] == VK_NULL_HANDLE )
        {
            continue;
        }
        if( id< MTLBuffer > ab = EncodeSet( sets[ i ] ) )
        {
            [e setBuffer:ab offset:0 atIndex:i];
        }
    }
    if( push && pushSize > 0 )
    {
        // Metal's struct of the push constants may be padded to 16 bytes; 'push' is 256
        [e setBytes:push length:PushLength( pushSize ) atIndex:MetalGen::PUSH_CONSTANT_BUFFER_INDEX];
    }
    [e dispatchThreadgroups:MTLSizeMake( groupsX, groupsY, groupsZ ) threadsPerThreadgroup:pi.threadsPerGroup];
    // the next pass reads what this one wrote
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
}

// a compute command on SectionCmd() that the native path can't do: skipped, and the generic
// compute passes go back to Vulkan from the next frame on
void SectionUnsupported( const char* what )
{
    std::lock_guard g( S().lock );
    S().computeBroken = true;
    LogOnce( std::string( what ) + " was recorded inside a native Metal section: not supported, "
                                   "the compute passes go back to MoltenVK" );
}

// the same for graphics / copies: the render passes go back to Vulkan from the next frame on
void RasterUnsupported( const std::string& what )
{
    std::lock_guard g( S().lock );
    S().rasterBroken = true;
    LogOnce( what + ": not supported in a native Metal section, the render passes go back to MoltenVK" );
}

void SectionBindPipeline( VkPipelineBindPoint bp, VkPipeline pipeline );
void SectionBindSets( VkPipelineBindPoint bp, uint32_t firstSet, uint32_t count, const VkDescriptorSet* sets );

void SectionPushConstants( VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data );

void SectionDispatch( uint32_t x, uint32_t y, uint32_t z )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( !s.cb )
        {
            LogOnce( "a dispatch on the section command buffer outside of a section" );
            return;
        }
        if( s.renderEnc )
        {
            RasterUnsupported( "a dispatch inside a render pass" );
            return;
        }

        auto pit = s.pipelines.find( s.bound.pipeline );
        if( pit == s.pipelines.end() )
        {
            s.computeBroken = true;
            LogOnce( "a dispatch without a known compute pipeline in a native Metal section" );
            return;
        }
        if( !pit->second.mtl && !pit->second.failed )
        {
            // e.g. created by a shader reload: compiled now (a hitch, once)
            if( pit->second.name.empty() )
            {
                pit->second.name = DefaultPipelineName( pit->first );
            }
            PreparePipelineLocked( pit->first, true );
            pit = s.pipelines.find( s.bound.pipeline );
        }
        if( pit == s.pipelines.end() || !pit->second.mtl )
        {
            s.computeBroken = true;
            LogOnce( "a compute pass couldn't be prepared for Metal, the compute passes go back to MoltenVK" );
            return;
        }

        uint32_t setCount = 0;
        auto     lit      = s.pipelineLayouts.find( pit->second.layout );
        if( lit != s.pipelineLayouts.end() )
        {
            setCount = std::min< uint32_t >( uint32_t( lit->second.size() ), MAX_BOUND_SETS );
        }
        EncodeDispatch( pit->second, s.bound.sets, setCount, s.bound.push, s.bound.pushSize, x, y, z );
    }
}

}

// ================================================================================================
// graphics in a section: render passes, pipelines, draws

namespace
{

constexpr uint32_t MAX_VERTEX_BINDINGS = 8;

struct GraphicsBound
{
    VkPipeline      pipeline                  = VK_NULL_HANDLE;
    VkDescriptorSet sets[ MAX_BOUND_SETS ]    = {};
    id< MTLBuffer > vb[ MAX_VERTEX_BINDINGS ];
    NSUInteger      vbOffset[ MAX_VERTEX_BINDINGS ] = {};
    id< MTLBuffer > ib;
    NSUInteger      ibOffset    = 0;
    MTLIndexType    ibType      = MTLIndexTypeUInt32;
    bool            hasViewport = false;
    bool            hasScissor  = false;
    VkViewport      viewport    = {};
    VkRect2D        scissor     = {};
    uint8_t         push[ 256 ] = {};
    uint32_t        pushSize    = 0;
};

struct RenderPassState
{
    MTLRenderPassDescriptor*           desc;
    std::vector< MTLPixelFormat >      colorFormats;
    std::vector< VkAttachmentStoreOp > colorStore;
    MTLPixelFormat                     depthFormat   = MTLPixelFormatInvalid;
    MTLPixelFormat                     stencilFormat = MTLPixelFormatInvalid;
    VkAttachmentStoreOp                depthStore    = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    uint32_t                           width = 0, height = 0;
    uint32_t                           viewCount = 1, firstView = 0;
    std::string                        formatKey;
    // a render pass the native path can't do: its draws are dropped
    bool skip = false;

    // what has to be set on the encoder before the next draw
    bool                  pipelineDirty = true;
    bool                  setsDirty     = true;
    bool                  vbDirty       = true;
    bool                  pushDirty     = true;
    bool                  viewportDirty = true;
    bool                  scissorDirty  = true;
    GraphicsPipelineInfo* gp            = nullptr;
    MTLPrimitiveType      prim          = MTLPrimitiveTypeTriangle;
};

struct GraphicsLibrary
{
    id< MTLLibrary > lib;
    std::string      entry;
};

struct Gfx
{
    GraphicsBound                                        bound;
    RenderPassState                                      rp;
    std::unordered_map< std::string, GraphicsLibrary >   libs;   // translated shader modules
    std::unordered_map< uint32_t, id< MTLDepthStencilState > > depthStates;
};

Gfx& G()
{
    static Gfx g;
    return g;
}

bool HasDepth( MTLPixelFormat f )
{
    return f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float ||
           f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatDepth24Unorm_Stencil8;
}

bool HasStencil( MTLPixelFormat f )
{
    return f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatDepth24Unorm_Stencil8 ||
           f == MTLPixelFormatStencil8 || f == MTLPixelFormatX32_Stencil8;
}

MTLLoadAction LoadAction( VkAttachmentLoadOp op )
{
    switch( op )
    {
        case VK_ATTACHMENT_LOAD_OP_LOAD: return MTLLoadActionLoad;
        case VK_ATTACHMENT_LOAD_OP_CLEAR: return MTLLoadActionClear;
        default: return MTLLoadActionDontCare;
    }
}

MTLVertexFormat VertexFormat( VkFormat f )
{
    switch( f )
    {
        case VK_FORMAT_R32_SFLOAT: return MTLVertexFormatFloat;
        case VK_FORMAT_R32G32_SFLOAT: return MTLVertexFormatFloat2;
        case VK_FORMAT_R32G32B32_SFLOAT: return MTLVertexFormatFloat3;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return MTLVertexFormatFloat4;
        case VK_FORMAT_R32_UINT: return MTLVertexFormatUInt;
        case VK_FORMAT_R32G32_UINT: return MTLVertexFormatUInt2;
        case VK_FORMAT_R32G32B32_UINT: return MTLVertexFormatUInt3;
        case VK_FORMAT_R32G32B32A32_UINT: return MTLVertexFormatUInt4;
        case VK_FORMAT_R32_SINT: return MTLVertexFormatInt;
        case VK_FORMAT_R32G32_SINT: return MTLVertexFormatInt2;
        case VK_FORMAT_R32G32B32_SINT: return MTLVertexFormatInt3;
        case VK_FORMAT_R32G32B32A32_SINT: return MTLVertexFormatInt4;
        case VK_FORMAT_R16G16_SFLOAT: return MTLVertexFormatHalf2;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return MTLVertexFormatHalf4;
        case VK_FORMAT_R8G8B8A8_UNORM: return MTLVertexFormatUChar4Normalized;
        case VK_FORMAT_R8G8B8A8_SNORM: return MTLVertexFormatChar4Normalized;
        case VK_FORMAT_R8G8B8A8_UINT: return MTLVertexFormatUChar4;
        case VK_FORMAT_R8G8B8A8_SINT: return MTLVertexFormatChar4;
        case VK_FORMAT_B8G8R8A8_UNORM: return MTLVertexFormatUChar4Normalized_BGRA;
        default: return MTLVertexFormatInvalid;
    }
}

MTLBlendFactor BlendFactor( VkBlendFactor f )
{
    switch( f )
    {
        case VK_BLEND_FACTOR_ZERO: return MTLBlendFactorZero;
        case VK_BLEND_FACTOR_ONE: return MTLBlendFactorOne;
        case VK_BLEND_FACTOR_SRC_COLOR: return MTLBlendFactorSourceColor;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case VK_BLEND_FACTOR_DST_COLOR: return MTLBlendFactorDestinationColor;
        case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case VK_BLEND_FACTOR_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case VK_BLEND_FACTOR_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        case VK_BLEND_FACTOR_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
        case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
        case VK_BLEND_FACTOR_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
        case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
        case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
        case VK_BLEND_FACTOR_SRC1_COLOR: return MTLBlendFactorSource1Color;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return MTLBlendFactorOneMinusSource1Color;
        case VK_BLEND_FACTOR_SRC1_ALPHA: return MTLBlendFactorSource1Alpha;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return MTLBlendFactorOneMinusSource1Alpha;
        default: return MTLBlendFactorOne;
    }
}

MTLBlendOperation BlendOp( VkBlendOp op )
{
    switch( op )
    {
        case VK_BLEND_OP_SUBTRACT: return MTLBlendOperationSubtract;
        case VK_BLEND_OP_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
        case VK_BLEND_OP_MIN: return MTLBlendOperationMin;
        case VK_BLEND_OP_MAX: return MTLBlendOperationMax;
        default: return MTLBlendOperationAdd;
    }
}

MTLColorWriteMask WriteMask( VkColorComponentFlags m )
{
    // the bits are in the opposite order
    MTLColorWriteMask r = MTLColorWriteMaskNone;
    if( m & VK_COLOR_COMPONENT_R_BIT )
    {
        r |= MTLColorWriteMaskRed;
    }
    if( m & VK_COLOR_COMPONENT_G_BIT )
    {
        r |= MTLColorWriteMaskGreen;
    }
    if( m & VK_COLOR_COMPONENT_B_BIT )
    {
        r |= MTLColorWriteMaskBlue;
    }
    if( m & VK_COLOR_COMPONENT_A_BIT )
    {
        r |= MTLColorWriteMaskAlpha;
    }
    return r;
}

bool PrimitiveType( VkPrimitiveTopology t, MTLPrimitiveType& prim, MTLPrimitiveTopologyClass& cls )
{
    switch( t )
    {
        case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
            prim = MTLPrimitiveTypePoint;
            cls  = MTLPrimitiveTopologyClassPoint;
            return true;
        case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
            prim = MTLPrimitiveTypeLine;
            cls  = MTLPrimitiveTopologyClassLine;
            return true;
        case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
            prim = MTLPrimitiveTypeLineStrip;
            cls  = MTLPrimitiveTopologyClassLine;
            return true;
        case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
            prim = MTLPrimitiveTypeTriangle;
            cls  = MTLPrimitiveTopologyClassTriangle;
            return true;
        case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
            prim = MTLPrimitiveTypeTriangleStrip;
            cls  = MTLPrimitiveTopologyClassTriangle;
            return true;
        default: return false; // fans, adjacency, patches
    }
}

MTLViewport ToMtl( const VkViewport& v )
{
    MTLViewport m;
    m.originX = v.x;
    m.originY = v.y;
    m.width   = v.width;
    m.height  = v.height;
    m.znear   = v.minDepth;
    m.zfar    = v.maxDepth;
    return m;
}

// Metal requires the scissor to be inside the render target
MTLScissorRect ToMtl( const VkRect2D& r, uint32_t width, uint32_t height )
{
    const int64_t x0 = std::clamp< int64_t >( r.offset.x, 0, width );
    const int64_t y0 = std::clamp< int64_t >( r.offset.y, 0, height );
    const int64_t x1 = std::clamp< int64_t >( int64_t( r.offset.x ) + r.extent.width, x0, width );
    const int64_t y1 = std::clamp< int64_t >( int64_t( r.offset.y ) + r.extent.height, y0, height );
    MTLScissorRect m;
    m.x      = NSUInteger( x0 );
    m.y      = NSUInteger( y0 );
    m.width  = NSUInteger( x1 - x0 );
    m.height = NSUInteger( y1 - y0 );
    return m;
}

// under the lock
id< MTLFunction > GraphicsFunction( const ShaderStageInfo& st,
                                    VkPipelineLayout       layout,
                                    bool                   multiview,
                                    const std::string&     name,
                                    std::string&           error )
{
    State& s = S();
    char   key[ 64 ];
    snprintf( key, sizeof( key ), "%llx/%llx/%d",
              ( unsigned long long )( uintptr_t )st.module,
              ( unsigned long long )( uintptr_t )layout,
              multiview ? 1 : 0 );

    auto it = G().libs.find( key );
    if( it == G().libs.end() )
    {
        auto mit = s.modules.find( st.module );
        auto lit = s.pipelineLayouts.find( layout );
        if( mit == s.modules.end() || lit == s.pipelineLayouts.end() )
        {
            error = "shader module or pipeline layout unknown";
            return nil;
        }
        std::vector< MetalGen::SetSlots > sets;
        for( const auto& l : lit->second )
        {
            sets.push_back( l ? l->slots : MetalGen::SetSlots{} );
        }
        const MetalGen::ComputeShader sh = MetalGen::TranslateGraphics(
            mit->second.data(), mit->second.size(), sets, 30100, multiview );
        if( !sh.ok )
        {
            error = "SPIR-V to Metal translation failed: " + sh.error;
            return nil;
        }
        id< MTLLibrary > lib = CompileLibrary( s.mtlDevice, sh.msl, s.fastMath, name, error );
        if( !lib )
        {
            return nil;
        }
        it = G().libs.emplace( key, GraphicsLibrary{ lib, sh.entryPoint } ).first;
    }
    return MakeFunction( it->second.lib, it->second.entry, st.specEntries, st.specData, error );
}

// under the lock: the shaders and depth state of a graphics pipeline
bool PrepareGraphics( GraphicsPipelineInfo& gp )
{
    if( gp.vfn )
    {
        return true;
    }
    if( gp.failed )
    {
        return false;
    }
    @autoreleasepool
    {
        State&      s = S();
        std::string error;
        if( gp.name.empty() )
        {
            char buf[ 64 ];
            snprintf( buf, sizeof( buf ), "graphics pipeline %llx", ( unsigned long long )( uintptr_t )&gp );
            gp.name = buf;
        }
        const bool multiview = gp.viewCount > 1;

        id< MTLFunction > vfn = GraphicsFunction( gp.vs, gp.layout, multiview, gp.name + " (vertex)", error );
        id< MTLFunction > ffn;
        if( vfn && gp.hasFs )
        {
            ffn = GraphicsFunction( gp.fs, gp.layout, multiview, gp.name + " (fragment)", error );
        }
        if( !vfn || ( gp.hasFs && !ffn ) )
        {
            gp.failed = true;
            Log( "%s: %s", gp.name.c_str(), error.c_str() );
            return false;
        }

        const uint32_t dsKey = ( gp.depthTest ? 1u : 0u ) | ( gp.depthWrite ? 2u : 0u ) | ( uint32_t( gp.depthCompare ) << 4 );
        auto           dit   = G().depthStates.find( dsKey );
        if( dit == G().depthStates.end() )
        {
            MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
            dd.depthCompareFunction = gp.depthTest ? CompareFunction( gp.depthCompare ) : MTLCompareFunctionAlways;
            dd.depthWriteEnabled    = gp.depthWrite;
            dit = G().depthStates.emplace( dsKey, [s.mtlDevice newDepthStencilStateWithDescriptor:dd] ).first;
        }

        gp.vfn        = vfn;
        gp.ffn        = ffn;
        gp.depthState = dit->second;
        return true;
    }
}

// under the lock: the pipeline state for the attachments of the current render pass
id< MTLRenderPipelineState > PipelineVariant( GraphicsPipelineInfo& gp, const RenderPassState& rp )
{
    auto it = gp.variants.find( rp.formatKey );
    if( it != gp.variants.end() )
    {
        return it->second;
    }

    @autoreleasepool
    {
        MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
        pd.label                        = [NSString stringWithUTF8String:gp.name.c_str()];
        pd.vertexFunction               = gp.vfn;
        pd.fragmentFunction             = gp.ffn;
        pd.rasterSampleCount            = 1;

        MTLPrimitiveType          prim;
        MTLPrimitiveTopologyClass cls;
        if( !PrimitiveType( gp.topology, prim, cls ) )
        {
            Log( "%s: unsupported primitive topology %d", gp.name.c_str(), int( gp.topology ) );
            gp.variants[ rp.formatKey ] = nil;
            return nil;
        }
        pd.inputPrimitiveTopology = cls;

        MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
        for( const auto& a : gp.attributes )
        {
            const MTLVertexFormat f = VertexFormat( a.format );
            if( f == MTLVertexFormatInvalid || a.binding >= MAX_VERTEX_BINDINGS )
            {
                Log( "%s: unsupported vertex attribute format %d", gp.name.c_str(), int( a.format ) );
                gp.variants[ rp.formatKey ] = nil;
                return nil;
            }
            vd.attributes[ a.location ].format      = f;
            vd.attributes[ a.location ].offset      = a.offset;
            vd.attributes[ a.location ].bufferIndex = MetalGen::VERTEX_BUFFER_BASE + a.binding;
        }
        for( const auto& b : gp.bindings )
        {
            if( b.binding >= MAX_VERTEX_BINDINGS )
            {
                continue;
            }
            MTLVertexBufferLayoutDescriptor* l = vd.layouts[ MetalGen::VERTEX_BUFFER_BASE + b.binding ];
            l.stride       = b.stride;
            l.stepFunction = b.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? MTLVertexStepFunctionPerInstance
                                                                          : MTLVertexStepFunctionPerVertex;
            l.stepRate     = 1;
        }
        pd.vertexDescriptor = vd;

        for( size_t i = 0; i < rp.colorFormats.size(); i++ )
        {
            if( rp.colorFormats[ i ] == MTLPixelFormatInvalid )
            {
                continue;
            }
            MTLRenderPipelineColorAttachmentDescriptor* ca = pd.colorAttachments[ i ];
            ca.pixelFormat                                 = rp.colorFormats[ i ];
            if( i < gp.blends.size() )
            {
                const VkPipelineColorBlendAttachmentState& b = gp.blends[ i ];
                ca.blendingEnabled             = b.blendEnable;
                ca.sourceRGBBlendFactor        = BlendFactor( b.srcColorBlendFactor );
                ca.destinationRGBBlendFactor   = BlendFactor( b.dstColorBlendFactor );
                ca.rgbBlendOperation           = BlendOp( b.colorBlendOp );
                ca.sourceAlphaBlendFactor      = BlendFactor( b.srcAlphaBlendFactor );
                ca.destinationAlphaBlendFactor = BlendFactor( b.dstAlphaBlendFactor );
                ca.alphaBlendOperation         = BlendOp( b.alphaBlendOp );
                ca.writeMask                   = WriteMask( b.colorWriteMask );
            }
        }
        if( HasDepth( rp.depthFormat ) )
        {
            pd.depthAttachmentPixelFormat = rp.depthFormat;
        }
        if( HasStencil( rp.stencilFormat ) )
        {
            pd.stencilAttachmentPixelFormat = rp.stencilFormat;
        }

        NSError*                     err = nil;
        id< MTLRenderPipelineState > pso = [S().mtlDevice newRenderPipelineStateWithDescriptor:pd error:&err];
        if( !pso )
        {
            Log( "%s: render pipeline failed: %s", gp.name.c_str(),
                 err ? err.localizedDescription.UTF8String : "?" );
        }
        gp.variants[ rp.formatKey ] = pso;
        return pso;
    }
}

void BeginRenderEncoder()
{
    State&           s  = S();
    RenderPassState& rp = G().rp;

    s.renderEnc       = [s.cb renderCommandEncoderWithDescriptor:rp.desc];
    s.renderEnc.label = @"RTGL1 native render pass";
    [s.renderEnc waitForFence:s.passFence beforeStages:MTLRenderStageVertex];
    if( s.waitASFence )
    {
        [s.renderEnc waitForFence:s.fence beforeStages:MTLRenderStageVertex];
    }

    rp.pipelineDirty = true;
    rp.setsDirty     = true;
    rp.vbDirty       = true;
    rp.pushDirty     = true;
    rp.viewportDirty = true;
    rp.scissorDirty  = true;
    rp.gp            = nullptr;
}

// forceStore: keep everything, the render pass continues in another encoder
void EndRenderEncoder( bool forceStore )
{
    State&           s  = S();
    RenderPassState& rp = G().rp;
    if( !s.renderEnc )
    {
        return;
    }
    for( size_t i = 0; i < rp.colorFormats.size(); i++ )
    {
        if( rp.colorFormats[ i ] != MTLPixelFormatInvalid )
        {
            const bool store = forceStore || rp.colorStore[ i ] != VK_ATTACHMENT_STORE_OP_DONT_CARE;
            [s.renderEnc setColorStoreAction:store ? MTLStoreActionStore : MTLStoreActionDontCare atIndex:i];
        }
    }
    if( rp.depthFormat != MTLPixelFormatInvalid )
    {
        const bool store = forceStore || rp.depthStore != VK_ATTACHMENT_STORE_OP_DONT_CARE;
        [s.renderEnc setDepthStoreAction:store ? MTLStoreActionStore : MTLStoreActionDontCare];
    }
    if( rp.stencilFormat != MTLPixelFormatInvalid )
    {
        [s.renderEnc setStencilStoreAction:forceStore ? MTLStoreActionStore : MTLStoreActionDontCare];
    }
    [s.renderEnc updateFence:s.passFence afterStages:MTLRenderStageFragment];
    [s.renderEnc endEncoding];
    s.renderEnc = nil;
}

void SectionBeginRenderPass( const VkRenderPassBeginInfo* bi )
{
    @autoreleasepool
    {
        std::lock_guard  g( S().lock );
        State&           s  = S();
        RenderPassState& rp = G().rp;
        if( !s.cb )
        {
            return;
        }
        if( s.renderEnc )
        {
            EndRenderEncoder( false );
        }
        rp = {};

        auto rpi = s.renderPasses.find( bi->renderPass );
        auto fbi = s.framebuffers.find( bi->framebuffer );
        if( rpi == s.renderPasses.end() || fbi == s.framebuffers.end() || !rpi->second.supported )
        {
            rp.skip = true;
            RasterUnsupported( "a render pass (unknown, or with several subpasses / input or resolve attachments)" );
            return;
        }
        const RenderPassInfo&  info = rpi->second;
        const FramebufferInfo& fb   = fbi->second;

        EndNonRenderEncoder();

        MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
        std::string              key;
        for( size_t i = 0; i < info.colorRefs.size(); i++ )
        {
            const uint32_t ref = info.colorRefs[ i ];
            id< MTLTexture > tex = ref < fb.views.size() ? TextureForView( fb.views[ ref ] ) : nil;
            if( ref == VK_ATTACHMENT_UNUSED || !tex )
            {
                if( ref != VK_ATTACHMENT_UNUSED )
                {
                    rp.skip = true;
                    RasterUnsupported( "a render pass attachment without a Metal texture" );
                    return;
                }
                rp.colorFormats.push_back( MTLPixelFormatInvalid );
                rp.colorStore.push_back( VK_ATTACHMENT_STORE_OP_DONT_CARE );
                key += "-,";
                continue;
            }
            const RenderPassInfo::Attachment& att = info.attachments[ ref ];

            MTLRenderPassColorAttachmentDescriptor* ca = desc.colorAttachments[ i ];
            ca.texture                                 = tex;
            ca.loadAction                              = LoadAction( att.load );
            if( att.load == VK_ATTACHMENT_LOAD_OP_CLEAR && bi->pClearValues && ref < bi->clearValueCount )
            {
                const VkClearColorValue& c = bi->pClearValues[ ref ].color;
                switch( att.format )
                {
                    case VK_FORMAT_R32_UINT:
                    case VK_FORMAT_R32G32_UINT:
                    case VK_FORMAT_R32G32B32A32_UINT:
                    case VK_FORMAT_R16_UINT:
                    case VK_FORMAT_R16G16_UINT:
                    case VK_FORMAT_R16G16B16A16_UINT:
                    case VK_FORMAT_R8_UINT:
                    case VK_FORMAT_R8G8B8A8_UINT:
                        ca.clearColor = MTLClearColorMake( c.uint32[ 0 ], c.uint32[ 1 ], c.uint32[ 2 ], c.uint32[ 3 ] );
                        break;
                    case VK_FORMAT_R32_SINT:
                    case VK_FORMAT_R32G32_SINT:
                    case VK_FORMAT_R32G32B32A32_SINT:
                    case VK_FORMAT_R16_SINT:
                    case VK_FORMAT_R16G16_SINT:
                    case VK_FORMAT_R16G16B16A16_SINT:
                    case VK_FORMAT_R8_SINT:
                    case VK_FORMAT_R8G8B8A8_SINT:
                        ca.clearColor = MTLClearColorMake( c.int32[ 0 ], c.int32[ 1 ], c.int32[ 2 ], c.int32[ 3 ] );
                        break;
                    default:
                        ca.clearColor = MTLClearColorMake( c.float32[ 0 ], c.float32[ 1 ], c.float32[ 2 ], c.float32[ 3 ] );
                        break;
                }
            }
            ca.storeAction = MTLStoreActionUnknown; // set at the end

            rp.colorFormats.push_back( tex.pixelFormat );
            rp.colorStore.push_back( att.store );
            key += std::to_string( int( tex.pixelFormat ) ) + ",";
        }

        if( info.depthRef != VK_ATTACHMENT_UNUSED )
        {
            id< MTLTexture > tex = info.depthRef < fb.views.size() ? TextureForView( fb.views[ info.depthRef ] ) : nil;
            if( !tex )
            {
                rp.skip = true;
                RasterUnsupported( "a render pass depth attachment without a Metal texture" );
                return;
            }
            const RenderPassInfo::Attachment& att = info.attachments[ info.depthRef ];
            const MTLPixelFormat              f   = tex.pixelFormat;
            if( HasDepth( f ) )
            {
                desc.depthAttachment.texture    = tex;
                desc.depthAttachment.loadAction = LoadAction( att.load );
                if( att.load == VK_ATTACHMENT_LOAD_OP_CLEAR && bi->pClearValues && info.depthRef < bi->clearValueCount )
                {
                    desc.depthAttachment.clearDepth = bi->pClearValues[ info.depthRef ].depthStencil.depth;
                }
                desc.depthAttachment.storeAction = MTLStoreActionUnknown;
                rp.depthFormat                   = f;
                rp.depthStore                    = att.store;
            }
            if( HasStencil( f ) )
            {
                desc.stencilAttachment.texture     = tex;
                desc.stencilAttachment.loadAction  = MTLLoadActionClear;
                desc.stencilAttachment.storeAction = MTLStoreActionUnknown;
                rp.stencilFormat                   = f;
            }
            key += "d" + std::to_string( int( f ) );
        }

        rp.width                = fb.width;
        rp.height               = fb.height;
        desc.renderTargetWidth  = fb.width;
        desc.renderTargetHeight = fb.height;
        if( info.viewMask != 0 )
        {
            rp.viewCount                 = uint32_t( __builtin_popcount( info.viewMask ) );
            rp.firstView                 = uint32_t( __builtin_ctz( info.viewMask ) );
            desc.renderTargetArrayLength = rp.viewCount;
        }
        key += "v" + std::to_string( rp.viewCount );

        rp.desc      = desc;
        rp.formatKey = key;
        BeginRenderEncoder();
    }
}

void SectionEndRenderPass()
{
    std::lock_guard g( S().lock );
    if( G().rp.skip )
    {
        G().rp.skip = false;
        return;
    }
    EndRenderEncoder( false );
}

// under the lock: sets what the next draw needs; false if it can't be drawn
bool PrepareDraw()
{
    State&           s  = S();
    GraphicsBound&   b  = G().bound;
    RenderPassState& rp = G().rp;
    if( !s.renderEnc || rp.skip )
    {
        return false;
    }
    id< MTLRenderCommandEncoder > e = s.renderEnc;

    if( rp.pipelineDirty )
    {
        auto it = s.gpipelines.find( b.pipeline );
        if( it == s.gpipelines.end() )
        {
            RasterUnsupported( "a draw without a known graphics pipeline" );
            return false;
        }
        GraphicsPipelineInfo& gp = it->second;
        if( !PrepareGraphics( gp ) )
        {
            RasterUnsupported( "a graphics pipeline that couldn't be prepared for Metal" );
            return false;
        }
        if( gp.viewCount != rp.viewCount )
        {
            RasterUnsupported( "a graphics pipeline used with a render pass with a different view count" );
            return false;
        }
        id< MTLRenderPipelineState > pso = PipelineVariant( gp, rp );
        if( !pso )
        {
            RasterUnsupported( "a graphics pipeline state" );
            return false;
        }
        MTLPrimitiveTopologyClass cls;
        PrimitiveType( gp.topology, rp.prim, cls );

        [e setRenderPipelineState:pso];
        [e setDepthStencilState:gp.depthState];
        [e setCullMode:( gp.cullMode == VK_CULL_MODE_FRONT_BIT  ? MTLCullModeFront
                         : gp.cullMode == VK_CULL_MODE_BACK_BIT ? MTLCullModeBack
                                                                : MTLCullModeNone )];
        // with y flipped in the vertex shaders, triangles appear where they would in Vulkan,
        // with the same winding on screen
        [e setFrontFacingWinding:gp.frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE ? MTLWindingCounterClockwise
                                                                                 : MTLWindingClockwise];
        if( gp.depthBias )
        {
            [e setDepthBias:gp.depthBiasConstant slopeScale:gp.depthBiasSlope clamp:gp.depthBiasClamp];
        }
        else
        {
            [e setDepthBias:0 slopeScale:0 clamp:0];
        }
        [e setTriangleFillMode:gp.wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];

        if( !gp.dynamicViewport )
        {
            [e setViewport:ToMtl( gp.viewport )];
        }
        if( !gp.dynamicScissor )
        {
            [e setScissorRect:ToMtl( gp.scissor, rp.width, rp.height )];
        }

        rp.gp            = &gp;
        rp.pipelineDirty = false;
        rp.setsDirty     = true;
        rp.pushDirty     = true;
        rp.viewportDirty = true;
        rp.scissorDirty  = true;
    }
    GraphicsPipelineInfo& gp = *rp.gp;

    if( rp.viewportDirty && gp.dynamicViewport )
    {
        const VkViewport full = { 0, 0, float( rp.width ), float( rp.height ), 0, 1 };
        [e setViewport:ToMtl( b.hasViewport ? b.viewport : full )];
        rp.viewportDirty = false;
    }
    if( rp.scissorDirty && gp.dynamicScissor )
    {
        const VkRect2D full = { { 0, 0 }, { rp.width, rp.height } };
        [e setScissorRect:ToMtl( b.hasScissor ? b.scissor : full, rp.width, rp.height )];
        rp.scissorDirty = false;
    }
    if( rp.setsDirty )
    {
        auto lit = s.pipelineLayouts.find( gp.layout );
        const uint32_t setCount =
            lit != s.pipelineLayouts.end() ? std::min< uint32_t >( uint32_t( lit->second.size() ), MAX_BOUND_SETS ) : 0;
        for( uint32_t i = 0; i < setCount; i++ )
        {
            if( b.sets[ i ] == VK_NULL_HANDLE )
            {
                continue;
            }
            if( id< MTLBuffer > ab = EncodeSet( b.sets[ i ] ) )
            {
                [e setVertexBuffer:ab offset:0 atIndex:i];
                [e setFragmentBuffer:ab offset:0 atIndex:i];
            }
        }
        rp.setsDirty = false;
    }
    if( rp.pushDirty && b.pushSize > 0 )
    {
        [e setVertexBytes:b.push length:PushLength( b.pushSize ) atIndex:MetalGen::PUSH_CONSTANT_BUFFER_INDEX];
        [e setFragmentBytes:b.push length:PushLength( b.pushSize ) atIndex:MetalGen::PUSH_CONSTANT_BUFFER_INDEX];
        rp.pushDirty = false;
    }
    if( rp.vbDirty )
    {
        for( uint32_t i = 0; i < MAX_VERTEX_BINDINGS; i++ )
        {
            if( b.vb[ i ] )
            {
                [e setVertexBuffer:b.vb[ i ] offset:b.vbOffset[ i ] atIndex:MetalGen::VERTEX_BUFFER_BASE + i];
            }
        }
        rp.vbDirty = false;
    }
    if( rp.viewCount > 1 )
    {
        const uint32_t viewMask[ 2 ] = { rp.firstView, rp.viewCount };
        [e setVertexBytes:viewMask length:sizeof( viewMask ) atIndex:MetalGen::VIEW_MASK_BUFFER_INDEX];
        [e setFragmentBytes:viewMask length:sizeof( viewMask ) atIndex:MetalGen::VIEW_MASK_BUFFER_INDEX];
    }
    return true;
}

void SectionDraw( uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance )
{
    std::lock_guard g( S().lock );
    if( !PrepareDraw() || vertexCount == 0 || instanceCount == 0 )
    {
        return;
    }
    const RenderPassState& rp = G().rp;
    [S().renderEnc drawPrimitives:rp.prim
                      vertexStart:firstVertex
                      vertexCount:vertexCount
                    instanceCount:instanceCount * rp.viewCount
                     baseInstance:firstInstance];
}

void SectionDrawIndexed( uint32_t indexCount,
                         uint32_t instanceCount,
                         uint32_t firstIndex,
                         int32_t  vertexOffset,
                         uint32_t firstInstance )
{
    std::lock_guard g( S().lock );
    if( !PrepareDraw() || indexCount == 0 || instanceCount == 0 )
    {
        return;
    }
    const GraphicsBound&   b  = G().bound;
    const RenderPassState& rp = G().rp;
    if( !b.ib )
    {
        RasterUnsupported( "an indexed draw without an index buffer" );
        return;
    }
    const NSUInteger indexSize = b.ibType == MTLIndexTypeUInt16 ? 2 : 4;
    [S().renderEnc drawIndexedPrimitives:rp.prim
                              indexCount:indexCount
                               indexType:b.ibType
                             indexBuffer:b.ib
                       indexBufferOffset:b.ibOffset + NSUInteger( firstIndex ) * indexSize
                           instanceCount:instanceCount * rp.viewCount
                              baseVertex:vertexOffset
                            baseInstance:firstInstance];
}

// Metal has no draw count from a buffer: a small compute pass copies the commands, with the
// ones past the count made empty, and maxDrawCount indirect draws read them. That compute pass
// is done before the render pass if PrepareDrawIndexedIndirectCount was called with the same
// arguments; otherwise the render pass is paused for it and continues in a new encoder.

struct IndirectCountArgs
{
    VkBuffer     buffer      = VK_NULL_HANDLE;
    VkDeviceSize offset      = 0;
    VkBuffer     countBuffer = VK_NULL_HANDLE;
    VkDeviceSize countOffset = 0;
    uint32_t     maxDrawCount = 0;
    uint32_t     stride       = 0;

    bool operator==( const IndirectCountArgs& o ) const
    {
        return buffer == o.buffer && offset == o.offset && countBuffer == o.countBuffer &&
               countOffset == o.countOffset && maxDrawCount == o.maxDrawCount && stride == o.stride;
    }
};
IndirectCountArgs g_preparedIndirect; // compacted into indirectScratch, not drawn yet

constexpr NSUInteger INDIRECT_ARGS_SIZE = 20; // MTLDrawIndexedPrimitivesIndirectArguments

// under the lock, outside of a render pass
bool EncodeIndirectCompaction( const IndirectCountArgs& a )
{
    State& s = S();
    if( !s.compactIndirect )
    {
        return false;
    }
    id< MTLBuffer > src, cnt;
    NSUInteger      srcOffset = 0, cntOffset = 0;
    if( !ResolveBuffer( a.buffer, a.offset, src, srcOffset ) ||
        !ResolveBuffer( a.countBuffer, a.countOffset, cnt, cntOffset ) || ( srcOffset & 3 ) ||
        ( cntOffset & 3 ) || ( a.stride & 3 ) || a.stride < INDIRECT_ARGS_SIZE )
    {
        return false;
    }
    if( !s.indirectScratch || s.indirectScratch.length < a.maxDrawCount * INDIRECT_ARGS_SIZE )
    {
        s.indirectScratch =
            [s.mtlDevice newBufferWithLength:std::max< NSUInteger >( a.maxDrawCount * INDIRECT_ARGS_SIZE, 4096 )
                                     options:MTLResourceStorageModePrivate];
        s.indirectScratch.label = @"RTGL1 indirect draw commands";
    }

    id< MTLComputeCommandEncoder > c = ComputeEnc();
    [c setComputePipelineState:s.compactIndirect];
    [c setBuffer:src offset:srcOffset atIndex:0];
    [c setBuffer:cnt offset:cntOffset atIndex:1];
    [c setBuffer:s.indirectScratch offset:0 atIndex:2];
    const uint32_t params[ 4 ] = { a.stride, a.maxDrawCount, 0, 0 };
    [c setBytes:params length:sizeof( params ) atIndex:3];
    [c dispatchThreadgroups:MTLSizeMake( ( a.maxDrawCount + 63 ) / 64, 1, 1 )
        threadsPerThreadgroup:MTLSizeMake( 64, 1, 1 )];
    [c memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return true;
}

void SectionDrawIndexedIndirectCount( VkBuffer     buffer,
                                      VkDeviceSize offset,
                                      VkBuffer     countBuffer,
                                      VkDeviceSize countOffset,
                                      uint32_t     maxDrawCount,
                                      uint32_t     stride )
{
    @autoreleasepool
    {
        std::lock_guard  g( S().lock );
        State&           s  = S();
        GraphicsBound&   b  = G().bound;
        RenderPassState& rp = G().rp;

        const IndirectCountArgs args = { buffer, offset, countBuffer, countOffset, maxDrawCount, stride };
        const bool              prepared = g_preparedIndirect == args;
        g_preparedIndirect               = {};

        if( !s.renderEnc || rp.skip || maxDrawCount == 0 )
        {
            return;
        }
        if( rp.viewCount > 1 || !b.ib )
        {
            RasterUnsupported( "vkCmdDrawIndexedIndirectCount" );
            return;
        }

        if( !prepared )
        {
            EndRenderEncoder( true );
            const bool ok = EncodeIndirectCompaction( args );
            EndNonRenderEncoder();

            // continue the render pass where it was
            for( size_t i = 0; i < rp.colorFormats.size(); i++ )
            {
                if( rp.colorFormats[ i ] != MTLPixelFormatInvalid )
                {
                    rp.desc.colorAttachments[ i ].loadAction = MTLLoadActionLoad;
                }
            }
            if( rp.depthFormat != MTLPixelFormatInvalid )
            {
                rp.desc.depthAttachment.loadAction = MTLLoadActionLoad;
            }
            if( rp.stencilFormat != MTLPixelFormatInvalid )
            {
                rp.desc.stencilAttachment.loadAction = MTLLoadActionLoad;
            }
            BeginRenderEncoder();

            if( !ok )
            {
                RasterUnsupported( "vkCmdDrawIndexedIndirectCount with these buffers" );
                return;
            }
        }

        if( !PrepareDraw() )
        {
            return;
        }
        for( uint32_t i = 0; i < maxDrawCount; i++ )
        {
            [s.renderEnc drawIndexedPrimitives:rp.prim
                                     indexType:b.ibType
                                   indexBuffer:b.ib
                             indexBufferOffset:b.ibOffset
                                indirectBuffer:s.indirectScratch
                          indirectBufferOffset:i * INDIRECT_ARGS_SIZE];
        }
    }
}

void SectionCopyBuffer( VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t count, const VkBufferCopy* regions )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( !s.cb )
        {
            return;
        }
        if( s.renderEnc )
        {
            RasterUnsupported( "a buffer copy inside a render pass" );
            return;
        }
        for( uint32_t i = 0; i < count; i++ )
        {
            const VkBufferCopy& r = regions[ i ];
            id< MTLBuffer >     src, dst;
            NSUInteger          so = 0, dstOff = 0;
            if( !ResolveBuffer( srcBuffer, r.srcOffset, src, so ) || !ResolveBuffer( dstBuffer, r.dstOffset, dst, dstOff ) )
            {
                RasterUnsupported( "a buffer copy with buffers without Metal buffers" );
                return;
            }
            if( ( so & 3 ) || ( dstOff & 3 ) || ( r.size & 3 ) )
            {
                RasterUnsupported( "a buffer copy that isn't 4-byte aligned" );
                return;
            }
            if( r.size == 0 )
            {
                continue;
            }
            [BlitEnc() copyFromBuffer:src sourceOffset:so toBuffer:dst destinationOffset:dstOff size:r.size];
        }
    }
}

const char* COMPACT_INDIRECT_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct DrawArgs { uint indexCount; uint instanceCount; uint indexStart; int baseVertex; uint baseInstance; };
kernel void rgCompactIndirect( device const uchar* src   [[buffer(0)]],
                               device const uint*  count [[buffer(1)]],
                               device DrawArgs*    dst   [[buffer(2)]],
                               constant uint4&     p     [[buffer(3)]],
                               uint i [[thread_position_in_grid]] )
{
    if( i >= p.y ) return;
    DrawArgs a = { 0u, 0u, 0u, 0, 0u };
    if( i < count[ 0 ] )
    {
        device const uint* s = (device const uint*)( src + i * p.x );
        a.indexCount    = s[ 0 ];
        a.instanceCount = s[ 1 ];
        a.indexStart    = s[ 2 ];
        a.baseVertex    = int( s[ 3 ] );
        a.baseInstance  = s[ 4 ];
    }
    dst[ i ] = a;
}
)MSL";

// vkCmdBlitImage: scaled copy of a region with a sampler (the destination must be writable
// by shaders, as RTGL1's framebuffer images are)
const char* BLIT_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;
// as in Vulkan: destination pixel centers map linearly from dst0..dst1 to src0..src1
// (a region is flipped when an end is before its start)
struct BlitParams { int2 dst0; int2 dst1; float2 src0; float2 src1; float2 srcTexSize; };
kernel void rgBlit( texture2d< float, access::sample > src [[texture(0)]],
                    texture2d< float, access::write >  dst [[texture(1)]],
                    sampler                            smp [[sampler(0)]],
                    constant BlitParams&               p   [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]] )
{
    const int2 lo = min( p.dst0, p.dst1 );
    const int2 n  = abs( p.dst1 - p.dst0 );
    if( int( gid.x ) >= n.x || int( gid.y ) >= n.y ) return;
    const int2   d  = lo + int2( gid );
    const float2 t  = ( float2( d ) + 0.5 - float2( p.dst0 ) ) / float2( p.dst1 - p.dst0 );
    const float2 uv = ( p.src0 + t * ( p.src1 - p.src0 ) ) / p.srcTexSize;
    dst.write( src.sample( smp, uv, level( 0 ) ), uint2( d ) );
}
)MSL";

id< MTLComputePipelineState > HelperPipeline( const char* msl, NSString* entry, const char* name )
{
    State&            s = S();
    std::string       error;
    id< MTLLibrary >  lib = CompileLibrary( s.mtlDevice, msl, false, name, error );
    id< MTLFunction > fn  = lib ? [lib newFunctionWithName:entry] : nil;
    NSError*          err = nil;
    id< MTLComputePipelineState > ps = fn ? [s.mtlDevice newComputePipelineStateWithFunction:fn error:&err] : nil;
    if( !ps )
    {
        Log( "couldn't create the %s helper: %s", name,
             !error.empty() ? error.c_str() : ( err ? err.localizedDescription.UTF8String : "?" ) );
    }
    return ps;
}

// at Init
void CreateHelpers()
{
    State& s          = S();
    s.compactIndirect = HelperPipeline( COMPACT_INDIRECT_MSL, @"rgCompactIndirect", "indirect draw" );
    s.blitPipeline    = HelperPipeline( BLIT_MSL, @"rgBlit", "blit" );

    MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
    d.sAddressMode          = MTLSamplerAddressModeClampToEdge;
    d.tAddressMode          = MTLSamplerAddressModeClampToEdge;
    d.minFilter             = MTLSamplerMinMagFilterNearest;
    d.magFilter             = MTLSamplerMinMagFilterNearest;
    s.blitNearest           = [s.mtlDevice newSamplerStateWithDescriptor:d];
    d.minFilter             = MTLSamplerMinMagFilterLinear;
    d.magFilter             = MTLSamplerMinMagFilterLinear;
    s.blitLinear            = [s.mtlDevice newSamplerStateWithDescriptor:d];
}

void SectionBlitImage( VkImage srcImage, VkImage dstImage, uint32_t count, const VkImageBlit* regions, VkFilter filter )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( !s.cb )
        {
            return;
        }
        if( s.renderEnc || !s.blitPipeline )
        {
            RasterUnsupported( "vkCmdBlitImage" );
            return;
        }
        id< MTLTexture > src = TextureForImage( srcImage );
        id< MTLTexture > dst = TextureForImage( dstImage );
        if( !src || !dst )
        {
            RasterUnsupported( "vkCmdBlitImage with images without Metal textures" );
            return;
        }
        for( uint32_t i = 0; i < count; i++ )
        {
            const VkImageBlit& r = regions[ i ];
            if( r.srcSubresource.mipLevel != 0 || r.dstSubresource.mipLevel != 0 ||
                r.srcSubresource.baseArrayLayer != 0 || r.dstSubresource.baseArrayLayer != 0 ||
                r.srcOffsets[ 1 ].z - r.srcOffsets[ 0 ].z != 1 || r.dstOffsets[ 1 ].z - r.dstOffsets[ 0 ].z != 1 )
            {
                RasterUnsupported( "vkCmdBlitImage of mip levels / layers / 3D regions" );
                return;
            }
            struct
            {
                int32_t dst0[ 2 ];
                int32_t dst1[ 2 ];
                float   src0[ 2 ];
                float   src1[ 2 ];
                float   srcTexSize[ 2 ];
            } p = {
                { r.dstOffsets[ 0 ].x, r.dstOffsets[ 0 ].y },
                { r.dstOffsets[ 1 ].x, r.dstOffsets[ 1 ].y },
                { float( r.srcOffsets[ 0 ].x ), float( r.srcOffsets[ 0 ].y ) },
                { float( r.srcOffsets[ 1 ].x ), float( r.srcOffsets[ 1 ].y ) },
                { float( src.width ), float( src.height ) },
            };
            const uint32_t w = uint32_t( std::abs( p.dst1[ 0 ] - p.dst0[ 0 ] ) );
            const uint32_t h = uint32_t( std::abs( p.dst1[ 1 ] - p.dst0[ 1 ] ) );
            if( w == 0 || h == 0 )
            {
                continue;
            }

            id< MTLComputeCommandEncoder > e = ComputeEnc();
            [e setComputePipelineState:s.blitPipeline];
            [e setTexture:src atIndex:0];
            [e setTexture:dst atIndex:1];
            [e setSamplerState:filter == VK_FILTER_NEAREST ? s.blitNearest : s.blitLinear atIndex:0];
            [e setBytes:&p length:sizeof( p ) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake( ( w + 7 ) / 8, ( h + 7 ) / 8, 1 )
                threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
        }
    }
}

void ForgetGraphicsLibraries( uint64_t handle )
{
    for( auto it = G().libs.begin(); it != G().libs.end(); )
    {
        unsigned long long m = 0, l = 0;
        int                mv = 0;
        const bool         parsed = sscanf( it->first.c_str(), "%llx/%llx/%d", &m, &l, &mv ) == 3;
        it = parsed && ( m == handle || l == handle ) ? G().libs.erase( it ) : std::next( it );
    }
}

void ShutdownGraphics()
{
    State& s          = S();
    s.compactIndirect = nil;
    s.blitPipeline    = nil;
    s.blitNearest     = nil;
    s.blitLinear      = nil;
    s.indirectScratch = nil;
    s.renderEnc       = nil;
    s.blitEnc         = nil;
    for( auto& [ p, gp ] : s.gpipelines )
    {
        gp.vfn        = nil;
        gp.ffn        = nil;
        gp.depthState = nil;
        gp.variants.clear();
    }
    G().libs.clear();
    G().depthStates.clear();
    G().bound = {};
    G().rp    = {};
}

void SectionBindPipeline( VkPipelineBindPoint bp, VkPipeline pipeline )
{
    std::lock_guard g( S().lock );
    if( bp == VK_PIPELINE_BIND_POINT_GRAPHICS )
    {
        if( G().bound.pipeline != pipeline )
        {
            G().bound.pipeline    = pipeline;
            G().rp.pipelineDirty = true;
        }
        return;
    }
    if( bp != VK_PIPELINE_BIND_POINT_COMPUTE )
    {
        LogOnce( "unsupported pipeline bind point in a native Metal section" );
        return;
    }
    S().bound.pipeline = pipeline;
}

void SectionBindSets( VkPipelineBindPoint bp, uint32_t firstSet, uint32_t count, const VkDescriptorSet* sets )
{
    std::lock_guard g( S().lock );
    VkDescriptorSet* dst = bp == VK_PIPELINE_BIND_POINT_GRAPHICS ? G().bound.sets : S().bound.sets;
    for( uint32_t i = 0; i < count && firstSet + i < MAX_BOUND_SETS; i++ )
    {
        dst[ firstSet + i ] = sets[ i ];
    }
    if( bp == VK_PIPELINE_BIND_POINT_GRAPHICS )
    {
        G().rp.setsDirty = true;
    }
}

void SectionPushConstants( VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data )
{
    std::lock_guard g( S().lock );
    if( stages & VK_SHADER_STAGE_COMPUTE_BIT )
    {
        BoundState& b = S().bound;
        if( offset + size <= sizeof( b.push ) )
        {
            memcpy( b.push + offset, data, size );
            b.pushSize = std::max( b.pushSize, offset + size );
        }
        else
        {
            LogOnce( "push constants are too large for the native path" );
        }
    }
    if( stages & ( VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT ) )
    {
        GraphicsBound& b = G().bound;
        if( offset + size <= sizeof( b.push ) )
        {
            memcpy( b.push + offset, data, size );
            b.pushSize       = std::max( b.pushSize, offset + size );
            G().rp.pushDirty = true;
        }
        else
        {
            LogOnce( "push constants are too large for the native path" );
        }
    }
}

void SectionBindVertexBuffers( uint32_t first, uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets )
{
    std::lock_guard g( S().lock );
    GraphicsBound&  b = G().bound;
    for( uint32_t i = 0; i < count; i++ )
    {
        const uint32_t binding = first + i;
        if( binding >= MAX_VERTEX_BINDINGS )
        {
            continue;
        }
        id< MTLBuffer > buf;
        NSUInteger      off = 0;
        if( !ResolveBuffer( buffers[ i ], offsets ? offsets[ i ] : 0, buf, off ) )
        {
            RasterUnsupported( "a vertex buffer without a Metal buffer" );
            continue;
        }
        b.vb[ binding ]       = buf;
        b.vbOffset[ binding ] = off;
    }
    G().rp.vbDirty = true;
}

void SectionBindIndexBuffer( VkBuffer buffer, VkDeviceSize offset, VkIndexType type )
{
    std::lock_guard g( S().lock );
    GraphicsBound&  b = G().bound;
    id< MTLBuffer > buf;
    NSUInteger      off = 0;
    if( type != VK_INDEX_TYPE_UINT16 && type != VK_INDEX_TYPE_UINT32 )
    {
        RasterUnsupported( "an index type" );
        return;
    }
    if( !ResolveBuffer( buffer, offset, buf, off ) )
    {
        RasterUnsupported( "an index buffer without a Metal buffer" );
        return;
    }
    b.ib       = buf;
    b.ibOffset = off;
    b.ibType   = type == VK_INDEX_TYPE_UINT16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
}

void SectionSetViewport( const VkViewport& v )
{
    std::lock_guard g( S().lock );
    G().bound.viewport    = v;
    G().bound.hasViewport = true;
    G().rp.viewportDirty  = true;
}

void SectionSetScissor( const VkRect2D& r )
{
    std::lock_guard g( S().lock );
    G().bound.scissor    = r;
    G().bound.hasScissor = true;
    G().rp.scissorDirty  = true;
}

}

bool RTGL1::Metal::IsComputeReady()
{
    const State& s = S();
    return s.enabled && s.computeEnv && !s.computeBroken;
}

bool RTGL1::Metal::IsRasterReady()
{
    const State& s = S();
    return IsComputeReady() && s.rasterEnv && !s.rasterBroken;
}

void RTGL1::Metal::BeginSection( VkCommandBuffer& cmd )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        assert( !s.cb );

        // the Vulkan work so far signals the event...
        const uint64_t vulkanDone = ++s.timeline;
        s.cmdManager->SubmitTimeline( cmd, s.semaphore, 0, vulkanDone );
        cmd           = SectionCmd();
        s.sectionDone = vulkanDone + 1;
        s.bound       = {};
        G().bound     = {};
        G().rp        = {};

        // ...that the Metal work waits for on the GPU
        s.cb       = [s.queue commandBuffer];
        s.cb.label = @"RTGL1 native";
        [s.cb encodeWaitForEvent:s.event value:vulkanDone];
        if( s.profile )
        {
            // measured separately: the wait for the Vulkan work isn't part of any pass
            [s.cb commit];
            s.profWaitCb = s.cb;
            s.cb         = [s.queue commandBuffer];
            s.cb.label   = @"RTGL1 native (profiled)";
        }

        // the encoders of the section wait for these (s.fence) if there are any
        s.waitASFence = EncodeAccelerationStructureBuilds();
        if( s.waitASFence && s.profile )
        {
            SplitForProfiling( Intern( "AS top level" ) );
        }
    }
}

void RTGL1::Metal::Dispatch( VkPipeline             pipeline,
                             const VkDescriptorSet* sets,
                             uint32_t               setCount,
                             uint32_t               groupsX,
                             uint32_t               groupsY,
                             uint32_t               groupsZ )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        assert( s.cb && !s.renderEnc );

        auto pit = s.pipelines.find( pipeline );
        if( pit == s.pipelines.end() || !pit->second.mtl )
        {
            LogOnce( "a dispatch uses a pipeline that isn't prepared for Metal" );
            return;
        }
        EncodeDispatch( pit->second, sets, setCount, nullptr, 0, groupsX, groupsY, groupsZ );
    }
}

void RTGL1::Metal::PrepareDrawIndexedIndirectCount( VkCommandBuffer cmd,
                                                    VkBuffer        buffer,
                                                    VkDeviceSize    offset,
                                                    VkBuffer        countBuffer,
                                                    VkDeviceSize    countOffset,
                                                    uint32_t        maxDrawCount,
                                                    uint32_t        stride )
{
    if( !IsSectionCmd( cmd ) )
    {
        return;
    }
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        g_preparedIndirect = {};
        if( !S().cb || S().renderEnc || maxDrawCount == 0 )
        {
            return;
        }
        const IndirectCountArgs args = { buffer, offset, countBuffer, countOffset, maxDrawCount, stride };
        if( EncodeIndirectCompaction( args ) )
        {
            g_preparedIndirect = args;
        }
    }
}

bool RTGL1::Metal::ExternalEncoding( id< MTLCommandBuffer > __strong& cb, id< MTLFence > __strong& fence )
{
    std::lock_guard g( S().lock );
    State&          s = S();
    if( !s.cb || s.renderEnc )
    {
        return false;
    }
    EndNonRenderEncoder();
    CommitResidency();
    cb    = s.cb;
    fence = s.passFence;
    return true;
}

void RTGL1::Metal::Mark( const char* name )
{
    std::lock_guard g( S().lock );
    State&          s = S();
    if( s.profile && s.cb )
    {
        @autoreleasepool
        {
            SplitForProfiling( Intern( name ) );
        }
    }
}

void RTGL1::Metal::DrainTimings( const std::function< void( const char*, double ) >& fn )
{
    std::vector< std::pair< const char*, double > > t;
    {
        std::lock_guard g( S().timingLock );
        t.swap( S().timings );
    }
    for( const auto& [ name, ms ] : t )
    {
        fn( name, ms );
    }
}

void RTGL1::Metal::EndSection( VkCommandBuffer& cmd )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        assert( s.cb );

        if( s.renderEnc )
        {
            LogOnce( "a native Metal section ended inside a render pass" );
            EndRenderEncoder( false );
        }
        EndNonRenderEncoder();
        CommitResidency();
        s.waitASFence = false;

        const uint64_t metalDone = ++s.timeline;
        assert( metalDone == s.sectionDone );
        [s.cb encodeSignalEvent:s.event value:metalDone];
        [s.cb addCompletedHandler:^( id< MTLCommandBuffer > c ) {
            if( c.status == MTLCommandBufferStatusError && S().loggedErrors.fetch_add( 1 ) < 10 )
            {
                Log( "GPU error in the native Metal section: %s",
                     c.error ? c.error.localizedDescription.UTF8String : "?" );
            }
        }];

        if( s.profile )
        {
            s.profCbs.push_back( { s.cb, Intern( "(end of section)" ) } );
            auto                   list   = std::make_shared< std::vector< ProfiledCb > >( std::move( s.profCbs ) );
            id< MTLCommandBuffer > waitCb = s.profWaitCb;
            s.profCbs.clear();
            s.profWaitCb = nil;

            const auto  stats   = s.asStats;
            const char* nBuilds = Intern( "BLAS builds" );
            const char* nRefits = Intern( "BLAS refits" );
            const char* nInst   = Intern( "TLAS instances" );

            [s.cb addCompletedHandler:^( id< MTLCommandBuffer > ) {
                std::vector< std::pair< const char*, double > > out;
                out.emplace_back( nBuilds, double( stats.blasBuilds ) );
                out.emplace_back( nRefits, double( stats.blasRefits ) );
                out.emplace_back( nInst, double( stats.tlasInstances ) );
                double prevEnd = waitCb.GPUEndTime;
                for( const ProfiledCb& p : *list )
                {
                    if( p.cb.status != MTLCommandBufferStatusCompleted )
                    {
                        out.clear();
                        break;
                    }
                    const double start = std::max( double( p.cb.GPUStartTime ), prevEnd );
                    const double ms    = std::max( 0.0, double( p.cb.GPUEndTime ) - start ) * 1000.0;
                    prevEnd            = p.cb.GPUEndTime;
                    out.emplace_back( p.name, ms );
                }
                list->clear();
                std::lock_guard g( S().timingLock );
                S().timings.insert( S().timings.end(), out.begin(), out.end() );
            }];
        }
        s.asStats = {};

        [s.cb commit];
        s.lastCb = s.cb;
        s.cb     = nil;

        // the following Vulkan work waits for it
        s.cmdManager->AddTimelineWaitForNextSubmit( s.semaphore, metalDone );
        cmd = s.cmdManager->StartGraphicsCmd();
    }
}

// ================================================================================================
// command recording shims

void RgMtl_vkCmdBindPipeline( VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipeline pipeline )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionBindPipeline( bp, pipeline );
        return;
    }
    vkCmdBindPipeline( cmd, bp, pipeline );
}

void RgMtl_vkCmdBindDescriptorSets( VkCommandBuffer        cmd,
                                    VkPipelineBindPoint    bp,
                                    VkPipelineLayout       layout,
                                    uint32_t               firstSet,
                                    uint32_t               count,
                                    const VkDescriptorSet* sets,
                                    uint32_t               dynamicOffsetCount,
                                    const uint32_t*        dynamicOffsets )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        if( dynamicOffsetCount > 0 )
        {
            SectionUnsupported( "a descriptor set bind with dynamic offsets" );
            return;
        }
        SectionBindSets( bp, firstSet, count, sets );
        return;
    }
    vkCmdBindDescriptorSets( cmd, bp, layout, firstSet, count, sets, dynamicOffsetCount, dynamicOffsets );
}

void RgMtl_vkCmdPushConstants( VkCommandBuffer    cmd,
                               VkPipelineLayout   layout,
                               VkShaderStageFlags stages,
                               uint32_t           offset,
                               uint32_t           size,
                               const void*        data )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionPushConstants( stages, offset, size, data );
        return;
    }
    vkCmdPushConstants( cmd, layout, stages, offset, size, data );
}

void RgMtl_vkCmdDispatch( VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionDispatch( x, y, z );
        return;
    }
    vkCmdDispatch( cmd, x, y, z );
}

void RgMtl_vkCmdPipelineBarrier( VkCommandBuffer              cmd,
                                 VkPipelineStageFlags         src,
                                 VkPipelineStageFlags         dst,
                                 VkDependencyFlags            flags,
                                 uint32_t                     memCount,
                                 const VkMemoryBarrier*       mem,
                                 uint32_t                     bufCount,
                                 const VkBufferMemoryBarrier* buf,
                                 uint32_t                     imgCount,
                                 const VkImageMemoryBarrier*  img )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        return; // the work of a section runs in order
    }
    vkCmdPipelineBarrier( cmd, src, dst, flags, memCount, mem, bufCount, buf, imgCount, img );
}

void RgMtl_svkCmdPipelineBarrier2KHR( VkCommandBuffer cmd, const VkDependencyInfo* info )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        return; // the work of a section runs in order
    }
    RTGL1::svkCmdPipelineBarrier2KHR( cmd, info );
}

void RgMtl_vkCmdCopyBuffer( VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy* regions )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionCopyBuffer( src, dst, count, regions );
        return;
    }
    vkCmdCopyBuffer( cmd, src, dst, count, regions );
}

void RgMtl_vkCmdCopyBufferToImage( VkCommandBuffer          cmd,
                                   VkBuffer                 src,
                                   VkImage                  dst,
                                   VkImageLayout            layout,
                                   uint32_t                 count,
                                   const VkBufferImageCopy* regions )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        RasterUnsupported( "vkCmdCopyBufferToImage" );
        return;
    }
    vkCmdCopyBufferToImage( cmd, src, dst, layout, count, regions );
}

void RgMtl_vkCmdBlitImage( VkCommandBuffer    cmd,
                           VkImage            src,
                           VkImageLayout      srcLayout,
                           VkImage            dst,
                           VkImageLayout      dstLayout,
                           uint32_t           count,
                           const VkImageBlit* regions,
                           VkFilter           filter )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionBlitImage( src, dst, count, regions, filter );
        return;
    }
    vkCmdBlitImage( cmd, src, srcLayout, dst, dstLayout, count, regions, filter );
}

void RgMtl_vkCmdWriteTimestamp( VkCommandBuffer cmd, VkPipelineStageFlagBits stage, VkQueryPool pool, uint32_t query )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        return; // profiling in a section uses Metal::Mark
    }
    vkCmdWriteTimestamp( cmd, stage, pool, query );
}

void RgMtl_vkCmdResetQueryPool( VkCommandBuffer cmd, VkQueryPool pool, uint32_t first, uint32_t count )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        return;
    }
    vkCmdResetQueryPool( cmd, pool, first, count );
}

void RgMtl_vkCmdBeginRenderPass( VkCommandBuffer cmd, const VkRenderPassBeginInfo* info, VkSubpassContents contents )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionBeginRenderPass( info );
        return;
    }
    vkCmdBeginRenderPass( cmd, info, contents );
}

void RgMtl_vkCmdEndRenderPass( VkCommandBuffer cmd )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionEndRenderPass();
        return;
    }
    vkCmdEndRenderPass( cmd );
}

void RgMtl_vkCmdBindVertexBuffers( VkCommandBuffer     cmd,
                                   uint32_t            first,
                                   uint32_t            count,
                                   const VkBuffer*     buffers,
                                   const VkDeviceSize* offsets )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionBindVertexBuffers( first, count, buffers, offsets );
        return;
    }
    vkCmdBindVertexBuffers( cmd, first, count, buffers, offsets );
}

void RgMtl_vkCmdBindIndexBuffer( VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkIndexType type )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionBindIndexBuffer( buffer, offset, type );
        return;
    }
    vkCmdBindIndexBuffer( cmd, buffer, offset, type );
}

void RgMtl_vkCmdSetViewport( VkCommandBuffer cmd, uint32_t first, uint32_t count, const VkViewport* viewports )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        if( first == 0 && count > 0 )
        {
            SectionSetViewport( viewports[ 0 ] );
        }
        return;
    }
    vkCmdSetViewport( cmd, first, count, viewports );
}

void RgMtl_vkCmdSetScissor( VkCommandBuffer cmd, uint32_t first, uint32_t count, const VkRect2D* scissors )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        if( first == 0 && count > 0 )
        {
            SectionSetScissor( scissors[ 0 ] );
        }
        return;
    }
    vkCmdSetScissor( cmd, first, count, scissors );
}

void RgMtl_vkCmdDraw( VkCommandBuffer cmd, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionDraw( vertexCount, instanceCount, firstVertex, firstInstance );
        return;
    }
    vkCmdDraw( cmd, vertexCount, instanceCount, firstVertex, firstInstance );
}

void RgMtl_vkCmdDrawIndexed( VkCommandBuffer cmd,
                             uint32_t        indexCount,
                             uint32_t        instanceCount,
                             uint32_t        firstIndex,
                             int32_t         vertexOffset,
                             uint32_t        firstInstance )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionDrawIndexed( indexCount, instanceCount, firstIndex, vertexOffset, firstInstance );
        return;
    }
    vkCmdDrawIndexed( cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance );
}

void RgMtl_vkCmdDrawIndexedIndirectCount( VkCommandBuffer cmd,
                                          VkBuffer        buffer,
                                          VkDeviceSize    offset,
                                          VkBuffer        countBuffer,
                                          VkDeviceSize    countOffset,
                                          uint32_t        maxDrawCount,
                                          uint32_t        stride )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionDrawIndexedIndirectCount( buffer, offset, countBuffer, countOffset, maxDrawCount, stride );
        return;
    }
    vkCmdDrawIndexedIndirectCount( cmd, buffer, offset, countBuffer, countOffset, maxDrawCount, stride );
}

void RgMtl_vkCmdClearAttachments( VkCommandBuffer          cmd,
                                  uint32_t                 attachmentCount,
                                  const VkClearAttachment* attachments,
                                  uint32_t                 rectCount,
                                  const VkClearRect*       rects )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        RasterUnsupported( "vkCmdClearAttachments" );
        return;
    }
    vkCmdClearAttachments( cmd, attachmentCount, attachments, rectCount, rects );
}
