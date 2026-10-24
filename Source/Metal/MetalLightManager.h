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

#pragma once

// Objective-C++ only

#import <Metal/Metal.h>

#include "Common.h"
#include "Containers.h"
#include "LightDefs.h"
#include "Generated/ShaderCommonC.h"

#include <vector>

namespace RTGL1
{

class MetalBindingTable;

// The same as LightManager, with Metal buffers:
// a frame's lights are encoded to its buffer, the previous frame's buffer is the "Prev" one.
// TODO: light grid (LIGHT_GRID_ENABLED is 0 anyway)
class MetalLightManager
{
public:
    explicit MetalLightManager( id< MTLDevice > device );

    MetalLightManager( const MetalLightManager& other )            = delete;
    MetalLightManager& operator=( const MetalLightManager& other ) = delete;

    void PrepareForFrame( uint32_t frameIndex );
    void Add( uint32_t frameIndex, const LightCopy& light, const RgTransform* transform = nullptr );
    void SetLightstyles( const RgStartFrameInfo& params );

    // LightSources, LightSourcesPrev, LightSourcesIndexPrevToCur, LightSourcesIndexCurToPrev
    void Bind( MetalBindingTable& table, uint32_t frameIndex ) const;

    uint32_t GetLightCount() const { return regLightCount; }
    uint32_t GetLightCountPrev() const { return regLightCount_Prev; }
    uint32_t DoesDirectionalLightExist() const { return dirLightCount > 0 ? 1 : 0; }
    uint32_t GetLightIndexForShaders( uint32_t frameIndex, const uint64_t* pLightUniqueId ) const;

    // LightManager::TryGetVolumetricLight
    auto TryGetVolumetricLight( const RgFloat3D&                 cameraPos,
                                const std::vector< LightCopy >&  from,
                                const std::optional< uint64_t >& fallback ) const -> std::optional< uint64_t >;

private:
    LightArrayIndex GetIndex( const ShLightEncoded& encodedLight ) const;
    void            IncrementCount( const ShLightEncoded& encodedLight );
    void            AddInternal( uint32_t frameIndex, uint64_t uniqueId, const ShLightEncoded& encodedLight );
    void FillMatchPrev( uint32_t curFrameIndex, LightArrayIndex lightIndexInCurFrame, UniqueLightID uniqueID );

private:
    // a ring of 3: while the CPU writes the current frame's lights,
    // a frame in flight might still read the previous frame's ones as "Prev"
    static constexpr uint32_t LightsRingSize = MAX_FRAMES_IN_FLIGHT + 1;
    id< MTLBuffer >           lights[ LightsRingSize ];
    uint32_t                  lightsCur{ 0 };
    uint32_t                  lightsPrev{ LightsRingSize - 1 };

    id< MTLBuffer > prevToCurIndex[ MAX_FRAMES_IN_FLIGHT ];
    id< MTLBuffer > curToPrevIndex[ MAX_FRAMES_IN_FLIGHT ];

    uint32_t regLightCount{ 0 };
    uint32_t regLightCount_Prev{ 0 };
    uint32_t dirLightCount{ 0 };
    uint32_t dirLightCount_Prev{ 0 };

    rgl::unordered_map< UniqueLightID, LightArrayIndex > uniqueIDToArrayIndex[ MAX_FRAMES_IN_FLIGHT ];

    std::vector< uint8_t > lightstyles;
};

}
