// Copyright (c) 2026 RTGL1 Metal backend contributors
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

#include "MetalLightManager.h"

#include "MetalComputePipelines.h"

#include "LightEncoding.h"

#include <cstring>

RTGL1::MetalLightManager::MetalLightManager( id< MTLDevice > device )
{
    for( auto& b : lights )
    {
        b       = [device newBufferWithLength:sizeof( ShLightEncoded ) * LIGHT_ARRAY_MAX_SIZE
                                options:MTLResourceStorageModeShared];
        b.label = @"Lights";
    }
    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        prevToCurIndex[ i ] = [device newBufferWithLength:sizeof( uint32_t ) * LIGHT_ARRAY_MAX_SIZE
                                                  options:MTLResourceStorageModeShared];
        curToPrevIndex[ i ] = [device newBufferWithLength:sizeof( uint32_t ) * LIGHT_ARRAY_MAX_SIZE
                                                  options:MTLResourceStorageModeShared];
        prevToCurIndex[ i ].label = @"Lights prev to cur";
        curToPrevIndex[ i ].label = @"Lights cur to prev";
        memset( prevToCurIndex[ i ].contents, 0xFF, prevToCurIndex[ i ].length );
        memset( curToPrevIndex[ i ].contents, 0xFF, curToPrevIndex[ i ].length );
    }
}

void RTGL1::MetalLightManager::PrepareForFrame( uint32_t frameIndex )
{
    regLightCount_Prev = regLightCount;
    dirLightCount_Prev = dirLightCount;

    regLightCount = 0;
    dirLightCount = 0;

    lightsPrev = lightsCur;
    lightsCur  = ( lightsCur + 1 ) % LightsRingSize;

    memset( prevToCurIndex[ frameIndex ].contents,
            0xFF,
            sizeof( uint32_t ) * GetLightArrayEnd( regLightCount_Prev, dirLightCount_Prev ) );
    // no need to clear curToPrevIndex, as it'll be filled in the cur frame

    uniqueIDToArrayIndex[ frameIndex ].clear();
}

RTGL1::LightArrayIndex RTGL1::MetalLightManager::GetIndex( const ShLightEncoded& encodedLight ) const
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

void RTGL1::MetalLightManager::IncrementCount( const ShLightEncoded& encodedLight )
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

void RTGL1::MetalLightManager::AddInternal( uint32_t              frameIndex,
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

    auto* dst = static_cast< ShLightEncoded* >( lights[ lightsCur ].contents );
    memcpy( &dst[ index.GetArrayIndex() ], &encodedLight, sizeof( ShLightEncoded ) );

    FillMatchPrev( frameIndex, index, uniqueId );
    // must be unique
    assert( uniqueIDToArrayIndex[ frameIndex ].find( uniqueId ) == uniqueIDToArrayIndex[ frameIndex ].end() );
    // save index for the next frame
    uniqueIDToArrayIndex[ frameIndex ][ uniqueId ] = index;
}

void RTGL1::MetalLightManager::FillMatchPrev( uint32_t        curFrameIndex,
                                              LightArrayIndex lightIndexInCurFrame,
                                              UniqueLightID   uniqueID )
{
    uint32_t    prevFrame         = ( curFrameIndex + 1 ) % MAX_FRAMES_IN_FLIGHT;
    const auto& uniqueToPrevIndex = uniqueIDToArrayIndex[ prevFrame ];

    auto found = uniqueToPrevIndex.find( uniqueID );
    if( found == uniqueToPrevIndex.end() )
    {
        return;
    }

    LightArrayIndex lightIndexInPrevFrame = found->second;

    auto* prev2cur = static_cast< uint32_t* >( prevToCurIndex[ curFrameIndex ].contents );
    prev2cur[ lightIndexInPrevFrame.GetArrayIndex() ] = lightIndexInCurFrame.GetArrayIndex();

    auto* cur2prev = static_cast< uint32_t* >( curToPrevIndex[ curFrameIndex ].contents );
    cur2prev[ lightIndexInCurFrame.GetArrayIndex() ] = lightIndexInPrevFrame.GetArrayIndex();
}

// The same as LightManager::Add
void RTGL1::MetalLightManager::Add( uint32_t frameIndex, const LightCopy& light, const RgTransform* transform )
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
                AddInternal( frameIndex,
                             light.base.uniqueID,
                             EncodeAsDirectionalLight(
                                 lext, CalculateLightStyle( light.additional, lightstyles ), transform ) );
            },
            [ & ]( const RgLightSphericalEXT& lext ) {
                if( IsLightColorTooDim( lext ) )
                {
                    return;
                }
                AddInternal( frameIndex,
                             light.base.uniqueID,
                             EncodeAsSphereLight(
                                 lext, CalculateLightStyle( light.additional, lightstyles ), transform ) );
            },
            [ & ]( const RgLightSpotEXT& lext ) {
                if( IsLightColorTooDim( lext ) )
                {
                    return;
                }
                AddInternal( frameIndex,
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
                AddInternal( frameIndex,
                             light.base.uniqueID,
                             EncodeAsTriangleLight( lext,
                                                    unnormalizedNormal,
                                                    CalculateLightStyle( light.additional, lightstyles ),
                                                    transform ) );
#else
                debug::Error( "Polygonal / triangle lights are not supported" );
#endif
            },
        },
        light.extension );
}

void RTGL1::MetalLightManager::SetLightstyles( const RgStartFrameInfo& params )
{
    if( !params.pLightstyleValues8 || params.lightstyleValuesCount == 0 )
    {
        return;
    }
    auto values = std::span{ params.pLightstyleValues8, params.lightstyleValuesCount };
    lightstyles.assign( values.begin(), values.end() );
}

uint32_t RTGL1::MetalLightManager::GetLightIndexForShaders( uint32_t frameIndex, const uint64_t* pLightUniqueId ) const
{
    if( pLightUniqueId == nullptr )
    {
        return LIGHT_INDEX_NONE;
    }
    const auto f = uniqueIDToArrayIndex[ frameIndex ].find( UniqueLightID{ *pLightUniqueId } );
    if( f == uniqueIDToArrayIndex[ frameIndex ].end() )
    {
        return LIGHT_INDEX_NONE;
    }
    return f->second.GetArrayIndex();
}

void RTGL1::MetalLightManager::Bind( MetalBindingTable& table, uint32_t frameIndex ) const
{
    table.SetBuffer( "LightSources", lights[ lightsCur ] );
    table.SetBuffer( "LightSourcesPrev", lights[ lightsPrev ] );
    table.SetBuffer( "LightSourcesIndexPrevToCur", prevToCurIndex[ frameIndex ] );
    table.SetBuffer( "LightSourcesIndexCurToPrev", curToPrevIndex[ frameIndex ] );
}

auto RTGL1::MetalLightManager::TryGetVolumetricLight( const RgFloat3D&                 cameraPos,
                                                      const std::vector< LightCopy >&  from,
                                                      const std::optional< uint64_t >& fallback ) const
    -> std::optional< uint64_t >
{
    return RTGL1::TryGetVolumetricLight( cameraPos, from, fallback, lightstyles );
}
