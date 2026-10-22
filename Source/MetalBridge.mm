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
    }
    vkDestroyPipeline( device, pipeline, alloc );
}

// ================================================================================================
// native path

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
            const char* r = getenv( "RTGL1_METAL_AS_REBUILD_EVERY" );
            if( r && r[ 0 ] )
            {
                s.asRebuildEvery = uint32_t( std::clamp( atoi( r ), 1, 1000 ) );
            }
        }

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

        NSError*           err = nil;
        MTLCompileOptions* co  = [MTLCompileOptions new];
        co.languageVersion     = MTLLanguageVersion3_1;
        if( @available( macOS 15.0, * ) )
        {
            // fast math lets the compiler assume no Inf / NaN, so the shaders' isnan / isinf
            // checks (that keep bad values out of the reservoirs and the denoiser history)
            // could be removed; relaxed keeps the other optimizations but honors Inf / NaN
            co.mathMode = job.fastMath ? MTLMathModeFast : MTLMathModeRelaxed;
        }
        id< MTLLibrary > lib   = [device newLibraryWithSource:[NSString stringWithUTF8String:sh.msl.c_str()]
                                                    options:co
                                                      error:&err];
        if( !lib )
        {
            // retry without the optional ray query hint added by MetalGen
            const std::string hint = "    ip.assume_geometry_type(geometry_type::triangle);\n";
            std::string       plain = sh.msl;
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
                    Log( "%s: compiled without the triangle-only ray query hint", job.name.c_str() );
                }
            }
        }
        if( !lib )
        {
            NSString* path = [NSTemporaryDirectory() stringByAppendingPathComponent:
                                [NSString stringWithFormat:@"rtgl1-%s.metal", job.name.c_str()]];
            [[NSString stringWithUTF8String:sh.msl.c_str()] writeToFile:path
                                                             atomically:YES
                                                               encoding:NSUTF8StringEncoding
                                                                  error:nil];
            std::string msg = err ? err.localizedDescription.UTF8String : "?";
            r.error = "Metal compile failed (source saved to " + std::string( path.UTF8String ) +
                      "): " + msg.substr( 0, 3000 );
            return r;
        }

        NSString* entry = [NSString stringWithUTF8String:sh.entryPoint.c_str()];

        // specialization constants -> function constants, by the types the shader declares
        // (bool, int, uint, float; SPIRV-Cross keeps the constant_id as the index)
        MTLFunctionConstantValues* fc    = [MTLFunctionConstantValues new];
        id< MTLFunction >          probe = [lib newFunctionWithName:entry];
        if( !probe )
        {
            r.error = "entry point '" + sh.entryPoint + "' not found";
            return r;
        }
        for( MTLFunctionConstant* c in probe.functionConstantsDictionary.allValues )
        {
            for( const VkSpecializationMapEntry& se : job.specEntries )
            {
                if( se.constantID != c.index || se.offset + se.size > job.specData.size() )
                {
                    continue;
                }
                const uint8_t* src = job.specData.data() + se.offset;
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

        id< MTLFunction > fn = [lib newFunctionWithName:entry constantValues:fc error:&err];
        if( !fn )
        {
            r.error = "function '" + sh.entryPoint + "' failed: " +
                      ( err ? err.localizedDescription.UTF8String : "?" );
            return r;
        }

        MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
        pd.computeFunction               = fn;
        pd.label                         = [NSString stringWithUTF8String:job.name.c_str()];
        id< MTLComputePipelineState > ps = [device newComputePipelineStateWithDescriptor:pd
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

id< MTLComputeCommandEncoder > NewEncoder( id< MTLCommandBuffer > cb )
{
    id< MTLComputeCommandEncoder > e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    e.label                          = @"RTGL1 native";
    return e;
}

// profiling: ends the current command buffer, so its GPU time is the work since the last mark
void SplitForProfiling( const char* internedName )
{
    State& s = S();
    [s.encoder updateFence:s.passFence];
    [s.encoder endEncoding];
    CommitResidency();
    [s.cb commit];
    s.profCbs.push_back( { s.cb, internedName } );

    s.cb       = [s.queue commandBuffer];
    s.cb.label = @"RTGL1 native (profiled)";
    s.encoder  = NewEncoder( s.cb );
    [s.encoder waitForFence:s.passFence];
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
    State& s = S();
    [s.encoder setComputePipelineState:pi.mtl];
    for( uint32_t i = 0; i < setCount; i++ )
    {
        if( sets[ i ] == VK_NULL_HANDLE )
        {
            continue;
        }
        if( id< MTLBuffer > ab = EncodeSet( sets[ i ] ) )
        {
            [s.encoder setBuffer:ab offset:0 atIndex:i];
        }
    }
    if( push && pushSize > 0 )
    {
        [s.encoder setBytes:push length:pushSize atIndex:MetalGen::PUSH_CONSTANT_BUFFER_INDEX];
    }
    [s.encoder dispatchThreadgroups:MTLSizeMake( groupsX, groupsY, groupsZ )
              threadsPerThreadgroup:pi.threadsPerGroup];
    // the next pass reads what this one wrote
    [s.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
}

// a command on SectionCmd() that the native path can't do: skipped, and the generic compute
// passes go back to Vulkan from the next frame on
void SectionUnsupported( const char* what )
{
    std::lock_guard g( S().lock );
    S().computeBroken = true;
    LogOnce( std::string( what ) + " was recorded inside a native Metal section: not supported, "
                                   "the compute passes go back to MoltenVK" );
}

void SectionBindPipeline( VkPipelineBindPoint bp, VkPipeline pipeline )
{
    if( bp != VK_PIPELINE_BIND_POINT_COMPUTE )
    {
        SectionUnsupported( "a graphics pipeline bind" );
        return;
    }
    std::lock_guard g( S().lock );
    S().bound.pipeline = pipeline;
}

void SectionBindSets( VkPipelineBindPoint bp, uint32_t firstSet, uint32_t count, const VkDescriptorSet* sets )
{
    if( bp != VK_PIPELINE_BIND_POINT_COMPUTE )
    {
        SectionUnsupported( "a graphics descriptor set bind" );
        return;
    }
    std::lock_guard g( S().lock );
    for( uint32_t i = 0; i < count && firstSet + i < MAX_BOUND_SETS; i++ )
    {
        S().bound.sets[ firstSet + i ] = sets[ i ];
    }
}

void SectionPushConstants( uint32_t offset, uint32_t size, const void* data )
{
    std::lock_guard g( S().lock );
    BoundState&     b = S().bound;
    if( offset + size > sizeof( b.push ) )
    {
        LogOnce( "push constants are too large for the native path" );
        return;
    }
    memcpy( b.push + offset, data, size );
    b.pushSize = std::max( b.pushSize, offset + size );
}

void SectionDispatch( uint32_t x, uint32_t y, uint32_t z )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( !s.encoder )
        {
            LogOnce( "a dispatch on the section command buffer outside of a section" );
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

bool RTGL1::Metal::IsComputeReady()
{
    const State& s = S();
    return s.enabled && s.computeEnv && !s.computeBroken;
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

        const bool built = EncodeAccelerationStructureBuilds();

        s.encoder = NewEncoder( s.cb );
        if( built )
        {
            [s.encoder waitForFence:s.fence];
            if( s.profile )
            {
                SplitForProfiling( Intern( "AS top level" ) );
            }
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
        assert( s.encoder );

        auto pit = s.pipelines.find( pipeline );
        if( pit == s.pipelines.end() || !pit->second.mtl )
        {
            LogOnce( "a dispatch uses a pipeline that isn't prepared for Metal" );
            return;
        }
        EncodeDispatch( pit->second, sets, setCount, nullptr, 0, groupsX, groupsY, groupsZ );
    }
}

void RTGL1::Metal::Mark( const char* name )
{
    std::lock_guard g( S().lock );
    State&          s = S();
    if( s.profile && s.cb && s.encoder )
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
        assert( s.cb && s.encoder );

        [s.encoder endEncoding];
        s.encoder = nil;
        CommitResidency();

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

            const auto stats = s.asStats;
            s.asStats        = {};
            const char* nBuilds = Intern( "BLAS builds" );
            const char* nRefits = Intern( "BLAS refits" );
            const char* nInst   = Intern( "TLAS instances" );

            [s.cb addCompletedHandler:^( id< MTLCommandBuffer > ) {
                std::vector< std::pair< const char*, double > > out;
                out.emplace_back( nBuilds, double( stats.blasBuilds ) );
                out.emplace_back( nRefits, double( stats.blasRefits ) );
                out.emplace_back( nInst, double( stats.tlasInstances ) );
                double                                          prevEnd = waitCb.GPUEndTime;
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
        SectionPushConstants( offset, size, data );
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
        return; // the passes of a section run in order
    }
    vkCmdPipelineBarrier( cmd, src, dst, flags, memCount, mem, bufCount, buf, imgCount, img );
}

void RgMtl_svkCmdPipelineBarrier2KHR( VkCommandBuffer cmd, const VkDependencyInfo* info )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        return; // the passes of a section run in order
    }
    RTGL1::svkCmdPipelineBarrier2KHR( cmd, info );
}

void RgMtl_vkCmdCopyBuffer( VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy* regions )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        SectionUnsupported( "vkCmdCopyBuffer" );
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
        SectionUnsupported( "vkCmdCopyBufferToImage" );
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
        SectionUnsupported( "vkCmdBlitImage" );
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
        SectionUnsupported( "vkCmdResetQueryPool" );
        return;
    }
    vkCmdResetQueryPool( cmd, pool, first, count );
}

void RgMtl_vkCmdBeginRenderPass( VkCommandBuffer cmd, const VkRenderPassBeginInfo* info, VkSubpassContents contents )
{
    if( Metal::IsSectionCmd( cmd ) )
    {
        // can't be skipped safely: the draws that follow would go to Vulkan
        Log( "vkCmdBeginRenderPass inside a native Metal section" );
        abort();
    }
    vkCmdBeginRenderPass( cmd, info, contents );
}
