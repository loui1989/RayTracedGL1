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

#include "RTGL1/RTGL1.h"

#include "ScratchImmediate.h"

#include <memory>
#include <string_view>

namespace RTGL1
{

// defined in CubemapManager.h
struct RgOriginalCubemapInfo;

// Native Metal backend: the same public interface as VulkanDevice,
// so RTGL1.cpp can use either one. Objective-C objects live in Impl (MetalDevice.mm),
// to keep this header plain C++.
class MetalDevice
{
public:
    explicit MetalDevice( const RgInstanceCreateInfo* pInfo );
    ~MetalDevice();

    MetalDevice( const MetalDevice& other )                = delete;
    MetalDevice( MetalDevice&& other ) noexcept            = delete;
    MetalDevice& operator=( const MetalDevice& other )     = delete;
    MetalDevice& operator=( MetalDevice&& other ) noexcept = delete;

    void UploadMeshPrimitive( const RgMeshInfo* pMesh, const RgMeshPrimitiveInfo* pPrimitive );
    void UploadLensFlare( const RgLensFlareInfo* pInfo );
    void SpawnFluid( const RgSpawnFluidInfo* pInfo );

    void UploadCamera( const RgCameraInfo* pInfo );

    void UploadLight( const RgLightInfo* pInfo );

    void ProvideOriginalTexture( const RgOriginalTextureInfo* pInfo );
    void ProvideOriginalCubemapTexture( const RgOriginalCubemapInfo* pInfo );
    void MarkOriginalTextureAsDeleted( const char* pTextureName );

    void StartFrame( const RgStartFrameInfo* pInfo );
    void DrawFrame( const RgDrawFrameInfo* pInfo );


    bool IsUpscaleTechniqueAvailable( RgRenderUpscaleTechnique technique,
                                      RgFrameGenerationMode    frameGeneration,
                                      const char**             ppFailureReason ) const;

    bool              IsDXGIAvailable( const char** ppFailureReason ) const;
    RgFeatureFlags    GetSupportedFeatures() const;
    RgUtilMemoryUsage RequestMemoryUsage() const;

    RgPrimitiveVertex* ScratchAllocForVertices( uint32_t count );
    void               ScratchFree( const RgPrimitiveVertex* pPointer );
    ScratchImmediate&  ScratchIm() { return scratchImmediate; }


    void Print( std::string_view msg, RgMessageSeverityFlags severity ) const;
    bool IsDevMode() const { return false; }

private:
    struct Impl;
    std::unique_ptr< Impl > impl;

    ScratchImmediate scratchImmediate;
};

}
