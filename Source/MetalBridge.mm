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

constexpr uint32_t ARG_BUFFER_RING = 4;

struct SetInfo
{
    std::shared_ptr< LayoutInfo >                          layout;
    VkDescriptorPool                                       pool = VK_NULL_HANDLE;
    std::unordered_map< uint32_t, std::vector< Descriptor > > descs;
    uint64_t                                               version = 1;

    id< MTLBuffer > argBuffers[ ARG_BUFFER_RING ];
    uint32_t        argBufferCur        = 0;
    uint64_t        encodedVersion      = 0;
    uint64_t        encodedASGeneration = 0;
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
};

struct MtlAS
{
    id< MTLAccelerationStructure > as;
    NSUInteger                     size = 0;
    id< MTLBuffer >                instances; // top level: instance descriptors
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
    if( upToDate && si.argBuffers[ si.argBufferCur ] )
    {
        return si.argBuffers[ si.argBufferCur ];
    }

    // a new buffer of the ring: the previous ones may still be read by the GPU
    si.argBufferCur        = ( si.argBufferCur + 1 ) % ARG_BUFFER_RING;
    const NSUInteger bytes = std::max< NSUInteger >( 1, layout.slots.totalSlots ) * sizeof( uint64_t );
    id< MTLBuffer > __strong& ab    = si.argBuffers[ si.argBufferCur ];
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
            const MTLAccelerationStructureSizes sizes = [s.mtlDevice accelerationStructureSizesWithDescriptor:d];

            MtlAS&     m         = s.as[ b.dst ];
            const bool recreated = EnsureAS( m, sizes.accelerationStructureSize );
            if( !m.as )
            {
                continue;
            }
            const bool refit = b.update && !recreated && ( d.usage & MTLAccelerationStructureUsageRefit );

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
        if( !m.instances || m.instances.length < instBytes )
        {
            m.instances       = [s.mtlDevice newBufferWithLength:instBytes options:MTLResourceStorageModeShared];
            m.instances.label = @"RTGL1 TLAS instances";
        }
        auto* dst = static_cast< MTLAccelerationStructureUserIDInstanceDescriptor* >( m.instances.contents );

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
        d.instanceDescriptorBuffer        = m.instances;
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
        if( !s.queue || !s.event || !s.fence || !s.blasFence )
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

        s.enabled = true;
        Log( "native path tracing enabled (RTGL1_METAL_RT=1) on %s", s.mtlDevice.name.UTF8String );
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
            for( auto& b : si.argBuffers )
            {
                b = nil;
            }
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
        s.dummyBlas  = nil;
        s.event      = nil;
        s.queue      = nil;
        s.mtlDevice  = nil;
        s.cmdManager = nullptr;
        s.enabled    = false;
    }
}

bool RTGL1::Metal::PreparePipeline( VkPipeline pipeline, const char* debugName )
{
    @autoreleasepool
    {
        std::lock_guard g( S().lock );
        State&          s = S();
        if( !s.enabled )
        {
            return false;
        }

        auto pit = s.pipelines.find( pipeline );
        if( pit == s.pipelines.end() )
        {
            Log( "%s: unknown pipeline", debugName );
            return false;
        }
        PipelineInfo& pi = pit->second;
        pi.name          = debugName ? debugName : "?";

        auto mit = s.modules.find( pi.module );
        auto lit = s.pipelineLayouts.find( pi.layout );
        if( mit == s.modules.end() || lit == s.pipelineLayouts.end() )
        {
            Log( "%s: shader module or pipeline layout unknown", pi.name.c_str() );
            return false;
        }

        std::vector< MetalGen::SetSlots > sets;
        for( const auto& l : lit->second )
        {
            sets.push_back( l ? l->slots : MetalGen::SetSlots{} );
        }

        const MetalGen::ComputeShader sh =
            MetalGen::TranslateCompute( mit->second.data(), mit->second.size(), sets, 30100 );
        if( !sh.ok )
        {
            Log( "%s: SPIR-V to Metal translation failed: %s", pi.name.c_str(), sh.error.c_str() );
            return false;
        }

        NSError*           err = nil;
        MTLCompileOptions* co  = [MTLCompileOptions new];
        co.languageVersion     = MTLLanguageVersion3_1;
        id< MTLLibrary > lib   = [s.mtlDevice newLibraryWithSource:[NSString stringWithUTF8String:sh.msl.c_str()]
                                                         options:co
                                                           error:&err];
        if( !lib )
        {
            NSString* path = [NSTemporaryDirectory() stringByAppendingPathComponent:
                                [NSString stringWithFormat:@"rtgl1-%s.metal", pi.name.c_str()]];
            [[NSString stringWithUTF8String:sh.msl.c_str()] writeToFile:path
                                                             atomically:YES
                                                               encoding:NSUTF8StringEncoding
                                                                  error:nil];
            std::string msg = err ? err.localizedDescription.UTF8String : "?";
            Log( "%s: Metal compile failed (source saved to %s): %.3000s", pi.name.c_str(), path.UTF8String, msg.c_str() );
            return false;
        }

        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        for( const VkSpecializationMapEntry& se : pi.specEntries )
        {
            if( se.size == sizeof( uint32_t ) && se.offset + se.size <= pi.specData.size() )
            {
                uint32_t v;
                memcpy( &v, pi.specData.data() + se.offset, sizeof( v ) );
                [fc setConstantValue:&v type:MTLDataTypeUInt atIndex:se.constantID];
            }
        }
        id< MTLFunction > fn = [lib newFunctionWithName:[NSString stringWithUTF8String:sh.entryPoint.c_str()]
                                         constantValues:fc
                                                  error:&err];
        if( !fn )
        {
            Log( "%s: function '%s' failed: %s", pi.name.c_str(), sh.entryPoint.c_str(),
                 err ? err.localizedDescription.UTF8String : "?" );
            return false;
        }

        MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
        pd.computeFunction               = fn;
        pd.label                         = [NSString stringWithUTF8String:pi.name.c_str()];
        id< MTLComputePipelineState > ps = [s.mtlDevice newComputePipelineStateWithDescriptor:pd
                                                                                      options:MTLPipelineOptionNone
                                                                                   reflection:nil
                                                                                        error:&err];
        const NSUInteger threads = NSUInteger( sh.localSize[ 0 ] ) * sh.localSize[ 1 ] * sh.localSize[ 2 ];
        if( !ps || ps.maxTotalThreadsPerThreadgroup < threads )
        {
            Log( "%s: pipeline failed: %s", pi.name.c_str(),
                 ps ? "too many threads per group" : ( err ? err.localizedDescription.UTF8String : "?" ) );
            return false;
        }

        pi.mtl             = ps;
        pi.threadsPerGroup = MTLSizeMake( sh.localSize[ 0 ], sh.localSize[ 1 ], sh.localSize[ 2 ] );
        Log( "%s: ready", pi.name.c_str() );
        return true;
    }
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
        cmd = VK_NULL_HANDLE;

        // ...that the Metal work waits for on the GPU
        s.cb       = [s.queue commandBuffer];
        s.cb.label = @"RTGL1 native path tracing";
        [s.cb encodeWaitForEvent:s.event value:vulkanDone];

        const bool built = EncodeAccelerationStructureBuilds();

        s.encoder       = [s.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        s.encoder.label = @"RTGL1 path tracer";
        if( built )
        {
            [s.encoder waitForFence:s.fence];
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

        [s.encoder setComputePipelineState:pit->second.mtl];
        for( uint32_t i = 0; i < setCount; i++ )
        {
            if( id< MTLBuffer > ab = EncodeSet( sets[ i ] ) )
            {
                [s.encoder setBuffer:ab offset:0 atIndex:i];
            }
        }
        [s.encoder dispatchThreadgroups:MTLSizeMake( groupsX, groupsY, groupsZ )
                  threadsPerThreadgroup:pit->second.threadsPerGroup];
        // the next pass reads what this one wrote
        [s.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
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

        if( s.residencyDirty )
        {
            if( @available( macOS 15.0, * ) )
            {
                [( id< MTLResidencySet > )s.residency commit];
            }
            s.residencyDirty = false;
        }

        const uint64_t metalDone = ++s.timeline;
        [s.cb encodeSignalEvent:s.event value:metalDone];
        [s.cb addCompletedHandler:^( id< MTLCommandBuffer > c ) {
            if( c.status == MTLCommandBufferStatusError && S().loggedErrors.fetch_add( 1 ) < 10 )
            {
                Log( "GPU error in the native path tracing: %s",
                     c.error ? c.error.localizedDescription.UTF8String : "?" );
            }
        }];
        [s.cb commit];
        s.lastCb = s.cb;
        s.cb     = nil;

        // the following Vulkan work waits for it
        s.cmdManager->AddTimelineWaitForNextSubmit( s.semaphore, metalDone );
        cmd = s.cmdManager->StartGraphicsCmd();
    }
}
