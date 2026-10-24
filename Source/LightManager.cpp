// Copyright (c) 2021 Sultim Tsyrendashiev
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

#include "LightManager.h"
#include "LightEncoding.h"

#include <cmath>
#include <array>

#include "Generated/ShaderCommonC.h"
#include "CmdLabel.h"
#include "RgException.h"
#include "Utils.h"


namespace
{
static_assert( LIGHT_GRID_ENABLED_ == LIGHT_GRID_ENABLED, "Change LIGHT_GRID_ENABLED_" );
#if LIGHT_GRID_ENABLED
constexpr VkDeviceSize GRID_LIGHTS_COUNT =
    LIGHT_GRID_CELL_SIZE * ( LIGHT_GRID_SIZE_X * LIGHT_GRID_SIZE_Y * LIGHT_GRID_SIZE_Z );
#endif
}

RTGL1::LightManager::LightManager( VkDevice                            _device,
                                   std::shared_ptr< MemoryAllocator >& _allocator )
    : device( _device )
    , regLightCount( 0 )
    , regLightCount_Prev( 0 )
    , dirLightCount( 0 )
    , dirLightCount_Prev( 0 )
    , descSetLayout( VK_NULL_HANDLE )
    , descPool( VK_NULL_HANDLE )
    , descSets{}
    , needDescSetUpdate{}
{
    lightsBuffer = std::make_shared< AutoBuffer >( _allocator );
    lightsBuffer->Create( sizeof( ShLightEncoded ) * LIGHT_ARRAY_MAX_SIZE,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          "Lights buffer" );

    lightsBuffer_Prev.Init( *_allocator,
                            sizeof( ShLightEncoded ) * LIGHT_ARRAY_MAX_SIZE,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            "Lights buffer - prev" );

#if LIGHT_GRID_ENABLED
    for( auto& buf : initialLightsGrid )
    {
        buf.Init( *_allocator,
                  sizeof( ShLightInCell ) * GRID_LIGHTS_COUNT,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  "Lights grid" );
    }
#endif

    prevToCurIndex = std::make_shared< AutoBuffer >( _allocator );
    prevToCurIndex->Create( sizeof( uint32_t ) * LIGHT_ARRAY_MAX_SIZE,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            "Lights buffer - prev to cur" );

    curToPrevIndex = std::make_shared< AutoBuffer >( _allocator );
    curToPrevIndex->Create( sizeof( uint32_t ) * LIGHT_ARRAY_MAX_SIZE,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            "Lights buffer - cur to prev" );

    CreateDescriptors();
}

RTGL1::LightManager::~LightManager()
{
    vkDestroyDescriptorSetLayout( device, descSetLayout, nullptr );
    vkDestroyDescriptorPool( device, descPool, nullptr );
}

void RTGL1::LightManager::PrepareForFrame( VkCommandBuffer cmd, uint32_t frameIndex )
{
    regLightCount_Prev = regLightCount;
    dirLightCount_Prev = dirLightCount;

    regLightCount = 0;
    dirLightCount = 0;

    // TODO: similar system to just swap desc sets, instead of actual copying
    if( GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) > 0 )
    {
        VkBufferCopy info = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size      = GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) *
                    sizeof( ShLightEncoded ),
        };

        vkCmdCopyBuffer(
            cmd, lightsBuffer->GetDeviceLocal(), lightsBuffer_Prev.GetBuffer(), 1, &info );
    }

    memset( prevToCurIndex->GetMapped( frameIndex ),
            0xFF,
            sizeof( uint32_t ) * GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) );
    // no need to clear curToPrevIndex, as it'll be filled in the cur frame

    uniqueIDToArrayIndex[ frameIndex ].clear();
}

void RTGL1::LightManager::Reset()
{
    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        memset( prevToCurIndex->GetMapped( i ),
                0xFF,
                sizeof( uint32_t ) *
                    std::max( GetLightArrayEnd( regLightCount, dirLightCount ),
                              GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) ) );
        memset( curToPrevIndex->GetMapped( i ),
                0xFF,
                sizeof( uint32_t ) *
                    std::max( GetLightArrayEnd( regLightCount, dirLightCount ),
                              GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) ) );

        uniqueIDToArrayIndex[ i ].clear();
    }

    regLightCount_Prev = regLightCount = 0;
    dirLightCount_Prev = dirLightCount = 0;
}

RTGL1::LightArrayIndex RTGL1::LightManager::GetIndex( const ShLightEncoded& encodedLight ) const
{
    switch( encodedLight.lightType )
    {
        case LIGHT_TYPE_DIRECTIONAL:
            return LightArrayIndex{ LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET + dirLightCount };
        case LIGHT_TYPE_SPHERE:
        case LIGHT_TYPE_TRIANGLE:
        case LIGHT_TYPE_SPOT:
            return LightArrayIndex{ LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET + regLightCount };
        default: assert( 0 ); return LightArrayIndex{ 0 };
    }
}

void RTGL1::LightManager::IncrementCount( const ShLightEncoded& encodedLight )
{
    switch( encodedLight.lightType )
    {
        case LIGHT_TYPE_DIRECTIONAL: dirLightCount++; break;
        case LIGHT_TYPE_SPHERE:
        case LIGHT_TYPE_TRIANGLE:
        case LIGHT_TYPE_SPOT: regLightCount++; break;
        default: assert( 0 );
    }
}

void RTGL1::LightManager::AddInternal( uint32_t              frameIndex,
                                       uint64_t              uniqueId,
                                       const ShLightEncoded& encodedLight )
{
    if( GetLightArrayEnd( regLightCount, dirLightCount ) >= LIGHT_ARRAY_MAX_SIZE )
    {
        assert( 0 );
        return;
    }

    const LightArrayIndex index = GetIndex( encodedLight );
    IncrementCount( encodedLight );

    auto* dst = lightsBuffer->GetMappedAs< ShLightEncoded* >( frameIndex );
    memcpy( &dst[ index.GetArrayIndex() ], &encodedLight, sizeof( ShLightEncoded ) );


    FillMatchPrev( frameIndex, index, uniqueId );
    // must be unique
    assert( uniqueIDToArrayIndex[ frameIndex ].find( uniqueId ) ==
            uniqueIDToArrayIndex[ frameIndex ].end() );
    // save index for the next frame
    uniqueIDToArrayIndex[ frameIndex ][ uniqueId ] = index;
}

void RTGL1::LightManager::Add( uint32_t           frameIndex,
                               const LightCopy&   light,
                               const RgTransform* transform )
{
    std::visit(
        ext::overloaded{
            [ & ]( const RgLightDirectionalEXT& lext ) {
                if( IsLightColorTooDim( lext ) )
                {
                    return;
                }

                if( dirLightCount > 0 )
                {
                    debug::Error( "Only one directional light is allowed" );
                    return;
                }

                AddInternal(
                    frameIndex,
                    light.base.uniqueID,
                    EncodeAsDirectionalLight(
                        lext, CalculateLightStyle( light.additional, lightstyles ), transform ) );
            },
            [ & ]( const RgLightSphericalEXT& lext ) {
                // Doom64-RT probe. The lava light family reported "119 uploaded"
                // from the gzdoom side every frame while lighting nothing, so the
                // question is what actually crosses this boundary and survives.
                // Keyed on the id range so it costs one compare for every other
                // light in the game. WARNING, never Error: RT_Print turns an RTGL
                // error into a modal box that exits.
                const bool l_isLava = ( light.base.uniqueID >= ( 1ull << 47 ) &&
                                        light.base.uniqueID < ( 1ull << 48 ) );
                if( l_isLava )
                {
                    static uint32_t s_seen  = 0;
                    static uint32_t s_dim   = 0;
                    static uint32_t s_added = 0;
                    s_seen++;
                    if( IsLightColorTooDim( lext ) )
                    {
                        s_dim++;
                    }
                    else
                    {
                        s_added++;
                    }
                    if( ( s_seen % 600 ) == 1 )
                    {
                        const float area = float( RG_PI ) * std::max( MIN_SPHERE_RADIUS, lext.radius ) *
                                           std::max( MIN_SPHERE_RADIUS, lext.radius );
                        auto  c    = RTGL1::Utils::UnpackColor4DPacked32< RgFloat3D >( lext.color );
                        float mult = CalculateLightStyle( light.additional, lightstyles );
                        debug::Warning( "LAVA PROBE: seen={} dim={} added={} | pos=({:.2f} {:.2f} {:.2f}) m "
                                        "intensity={:.1f} radius={:.3f} area={:.4f} color=({:.2f} {:.2f} {:.2f}) "
                                        "mult={:.2f} -> radiance=({:.1f} {:.1f} {:.1f}) | regLightCount={} arrayEnd={}",
                                        s_seen, s_dim, s_added,
                                        lext.position.data[ 0 ], lext.position.data[ 1 ], lext.position.data[ 2 ],
                                        lext.intensity, lext.radius, area,
                                        c.data[ 0 ], c.data[ 1 ], c.data[ 2 ], mult,
                                        c.data[ 0 ] * lext.intensity / area * mult,
                                        c.data[ 1 ] * lext.intensity / area * mult,
                                        c.data[ 2 ] * lext.intensity / area * mult,
                                        regLightCount,
                                        GetLightArrayEnd( regLightCount, dirLightCount ) );
                    }
                }

                if( IsLightColorTooDim( lext ) )
                {
                    return;
                }

                AddInternal(
                    frameIndex,
                    light.base.uniqueID,
                    EncodeAsSphereLight(
                        lext, CalculateLightStyle( light.additional, lightstyles ), transform ) );
            },
            [ & ]( const RgLightSpotEXT& lext ) {
                if( IsLightColorTooDim( lext ) )
                {
                    return;
                }

                AddInternal(
                    frameIndex,
                    light.base.uniqueID,
                    EncodeAsSpotLight(
                        lext, CalculateLightStyle( light.additional, lightstyles ), transform ) );
            },
            [ & ]( const RgLightPolygonalEXT& lext ) {
#if TRIANGLE_LIGHTS
                if( IsLightColorTooDim( lext ) )
                {
                    return;
                }

                RgFloat3D unnormalizedNormal = Utils::GetUnnormalizedNormal( lext.positions );
                if( Utils::Dot( unnormalizedNormal.data, unnormalizedNormal.data ) <= 0.0f )
                {
                    return;
                }

                AddInternal(
                    frameIndex,
                    light.base.uniqueID,
                    EncodeAsTriangleLight( lext,
                                           unnormalizedNormal,
                                           CalculateLightStyle( light.additional, lightstyles ),
                                           transform ) );
#else
                // WARNING, not Error: debug::Error is FATAL in this project
                // (exits the game), and a light type the build lacks must
                // degrade to "that light is missing", never to a crash.
                // Triangle lights are a half-removed upstream feature: the
                // shader decode behind TRIANGLE_LIGHTS is a deliberate
                // #error landmine (Light.h "Refine decoding, as it's
                // obsolete"), so the flag cannot simply be flipped on --
                // see docs/plan-area-lights-mis.md in the parent repo.
                debug::Warning( "Polygonal / triangle light IGNORED: not supported by this "
                                "build (TRIANGLE_LIGHTS off; the shader path behind the "
                                "flag is an unfinished upstream remnant)" );
#endif
            },
        },
        light.extension );
}

void RTGL1::LightManager::SubmitForFrame( VkCommandBuffer cmd, uint32_t frameIndex )
{
    CmdLabel label( cmd, "Copying lights" );

    lightsBuffer->CopyFromStaging( cmd,
                                   frameIndex,
                                   sizeof( ShLightEncoded ) *
                                       GetLightArrayEnd( regLightCount, dirLightCount ) );

    prevToCurIndex->CopyFromStaging(
        cmd,
        frameIndex,
        sizeof( uint32_t ) * GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) );
    curToPrevIndex->CopyFromStaging(
        cmd, frameIndex, sizeof( uint32_t ) * GetLightArrayEnd( regLightCount, dirLightCount ) );

    // should be used when buffers changed
    if( needDescSetUpdate[ frameIndex ] )
    {
        UpdateDescriptors( frameIndex );
        needDescSetUpdate[ frameIndex ] = false;
    }
}

void RTGL1::LightManager::BarrierLightGrid( VkCommandBuffer cmd, uint32_t frameIndex )
{
#if LIGHT_GRID_ENABLED
    VkBufferMemoryBarrier2 barrier = {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .pNext         = nullptr,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        .srcQueueFamilyIndex = 0,
        .dstQueueFamilyIndex = 0,
        .buffer              = initialLightsGrid[ frameIndex ].GetBuffer(),
        .offset              = 0,
        .size                = VK_WHOLE_SIZE,
    };

    VkDependencyInfo dependency = {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &barrier,
    };

    svkCmdPipelineBarrier2KHR( cmd, &dependency );
#endif
}

VkDescriptorSetLayout RTGL1::LightManager::GetDescSetLayout() const
{
    return descSetLayout;
}

VkDescriptorSet RTGL1::LightManager::GetDescSet( uint32_t frameIndex ) const
{
    return descSets[ frameIndex ];
}

void RTGL1::LightManager::FillMatchPrev( uint32_t        curFrameIndex,
                                         LightArrayIndex lightIndexInCurFrame,
                                         UniqueLightID   uniqueID )
{
    uint32_t prevFrame = ( curFrameIndex + 1 ) % MAX_FRAMES_IN_FLIGHT;
    const rgl::unordered_map< UniqueLightID, LightArrayIndex >& uniqueToPrevIndex =
        uniqueIDToArrayIndex[ prevFrame ];

    auto found = uniqueToPrevIndex.find( uniqueID );
    if( found == uniqueToPrevIndex.end() )
    {
        return;
    }

    LightArrayIndex lightIndexInPrevFrame = found->second;

    auto* prev2cur = prevToCurIndex->GetMappedAs< uint32_t* >( curFrameIndex );
    prev2cur[ lightIndexInPrevFrame.GetArrayIndex() ] = lightIndexInCurFrame.GetArrayIndex();

    auto* cur2prev = curToPrevIndex->GetMappedAs< uint32_t* >( curFrameIndex );
    cur2prev[ lightIndexInCurFrame.GetArrayIndex() ] = lightIndexInPrevFrame.GetArrayIndex();
}

constexpr uint32_t BINDINGS[] = {
    BINDING_LIGHT_SOURCES,
    BINDING_LIGHT_SOURCES_PREV,
    BINDING_LIGHT_SOURCES_INDEX_PREV_TO_CUR,
    BINDING_LIGHT_SOURCES_INDEX_CUR_TO_PREV,
#if LIGHT_GRID_ENABLED
    BINDING_INITIAL_LIGHTS_GRID,
    BINDING_INITIAL_LIGHTS_GRID_PREV,
#endif
};

void RTGL1::LightManager::CreateDescriptors()
{
    {
        std::array< VkDescriptorSetLayoutBinding, std::size( BINDINGS ) > bindings = {};

        for( uint32_t i = 0; i < std::size( BINDINGS ); i++ )
        {
            uint32_t bnd = BINDINGS[ i ];
            assert( i == bnd );

            VkDescriptorSetLayoutBinding& b = bindings[ bnd ];
            b.binding                       = bnd;
            b.descriptorType                = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount               = 1;
            b.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo = {
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = bindings.size(),
            .pBindings    = bindings.data(),
        };

        VkResult r = vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &descSetLayout );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME( device,
                        descSetLayout,
                        VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                        "Light buffers Desc set layout" );
    }
    {
        VkDescriptorPoolSize poolSize = {
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = std::size( BINDINGS ) * MAX_FRAMES_IN_FLIGHT,
        };

        VkDescriptorPoolCreateInfo poolInfo = {
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets       = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = 1,
            .pPoolSizes    = &poolSize,
        };

        VkResult r = vkCreateDescriptorPool( device, &poolInfo, nullptr, &descPool );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME(
            device, descPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Light buffers Desc set pool" );
    }
    {
        VkDescriptorSetAllocateInfo allocInfo = {
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = descPool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &descSetLayout,
        };

        for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
        {
            VkResult r = vkAllocateDescriptorSets( device, &allocInfo, &descSets[ i ] );

            VK_CHECKERROR( r );
            SET_DEBUG_NAME(
                device, descSets[ i ], VK_OBJECT_TYPE_DESCRIPTOR_SET, "Light buffers Desc set" );
        }

        for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
        {
            UpdateDescriptors( i );
        }
    }
}

void RTGL1::LightManager::UpdateDescriptors( uint32_t frameIndex )
{
    const VkBuffer buffers[] = {
        lightsBuffer->GetDeviceLocal(),
        lightsBuffer_Prev.GetBuffer(),
        prevToCurIndex->GetDeviceLocal(),
        curToPrevIndex->GetDeviceLocal(),
#if LIGHT_GRID_ENABLED
        initialLightsGrid[ frameIndex ].GetBuffer(),
        initialLightsGrid[ Utils::GetPreviousByModulo( frameIndex, MAX_FRAMES_IN_FLIGHT ) ]
            .GetBuffer(),
#endif
    };
    static_assert( std::size( BINDINGS ) == std::size( buffers ) );

    std::array< VkDescriptorBufferInfo, std::size( BINDINGS ) > bufs = {};
    std::array< VkWriteDescriptorSet, std::size( BINDINGS ) >   wrts = {};

    for( uint32_t i = 0; i < std::size( BINDINGS ); i++ )
    {
        uint32_t bnd = BINDINGS[ i ];
        // 'buffers' should be actually a map (binding->buffer), but a plain array works too, if
        // this is true
        assert( i == bnd );

        bufs[ bnd ] = VkDescriptorBufferInfo{
            .buffer = buffers[ bnd ],
            .offset = 0,
            .range  = VK_WHOLE_SIZE,
        };

        wrts[ bnd ] = VkWriteDescriptorSet{
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = descSets[ frameIndex ],
            .dstBinding      = bnd,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &bufs[ bnd ],
        };
    }

    vkUpdateDescriptorSets( device, wrts.size(), wrts.data(), 0, nullptr );
}

uint32_t RTGL1::LightManager::GetLightCount() const
{
    return regLightCount;
}

uint32_t RTGL1::LightManager::GetLightCountPrev() const
{
    return regLightCount_Prev;
}


uint32_t RTGL1::LightManager::DoesDirectionalLightExist() const
{
    return dirLightCount > 0 ? 1 : 0;
}

uint32_t RTGL1::LightManager::GetLightIndexForShaders( uint32_t        frameIndex,
                                                       const uint64_t* pLightUniqueId ) const
{
    if( pLightUniqueId == nullptr )
    {
        return LIGHT_INDEX_NONE;
    }
    UniqueLightID uniqueId = { *pLightUniqueId };

    const auto f = uniqueIDToArrayIndex[ frameIndex ].find( uniqueId );
    if( f == uniqueIDToArrayIndex[ frameIndex ].end() )
    {
        return LIGHT_INDEX_NONE;
    }

    return f->second.GetArrayIndex();
}

auto RTGL1::LightManager::TryGetVolumetricLight( const RgFloat3D&                 cameraPos,
                                                 const std::vector< LightCopy >&  from,
                                                 const std::optional< uint64_t >& fallback ) const
    -> std::optional< uint64_t >
{
    return RTGL1::TryGetVolumetricLight( cameraPos, from, fallback, lightstyles );
}

void RTGL1::LightManager::SetLightstyles( const RgStartFrameInfo& params )
{
    if( !params.pLightstyleValues8 || params.lightstyleValuesCount == 0 )
    {
        return;
    }

    auto values = std::span{ params.pLightstyleValues8, params.lightstyleValuesCount };
    lightstyles.assign( values.begin(), values.end() );
}

static_assert( RTGL1::MAX_FRAMES_IN_FLIGHT == 2 );
