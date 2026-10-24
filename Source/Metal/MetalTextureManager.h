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

#include "RTGL1/RTGL1.h"

#include "Containers.h"
#include "ImageLoader.h"
#include "ImageLoaderDev.h"
#include "Material.h"
#include "SamplerManager.h"
#include "TextureOverrides.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace RTGL1
{

// The same as TextureManager: materials of TEXTURES_PER_MATERIAL_COUNT textures
// (albedo, occlusion-roughness-metallic, normal, emissive, height),
// with the overriding files from <ovrdFolder>/mat/<name><postfix>.ktx2.
// Shaders index one bindless table of { texture, sampler } by texture index.
// Index 0 (EMPTY_TEXTURE_INDEX, MATERIAL_NO_TEXTURE) is a white texture.
class MetalTextureManager
{
public:
    static constexpr uint32_t MaxTextureCount = 16384;

    // an element of the table, TextureRef in RTGL1.metal
    struct TableEntry
    {
        MTLResourceID texture;
        MTLResourceID sampler;
    };

    MetalTextureManager( id< MTLDevice >              device,
                         id< MTLCommandQueue >        queue,
                         std::filesystem::path        ovrdFolder,
                         RgTextureSwizzling           pbrSwizzling,
                         bool                         forceMinificationFilterLinear,
                         bool                         forceNormalMapFilterLinear );
    ~MetalTextureManager();

    MetalTextureManager( const MetalTextureManager& other )            = delete;
    MetalTextureManager& operator=( const MetalTextureManager& other ) = delete;

    // TextureManager::TryCreateMaterial
    void Provide( const RgOriginalTextureInfo& info );
    // TextureManager::TryDestroyMaterial
    void MarkAsDeleted( const char* pTextureName );

    bool TryCreateImportedMaterial( const std::string&                        materialName,
                                    std::span< const std::filesystem::path >  fullPaths,
                                    std::span< const SamplerManager::Handle > samplers,
                                    RgTextureSwizzling                        customPbrSwizzling,
                                    bool                                      isReplacement );
    void FreeAllImportedMaterials( bool freeReplacements );

    auto GetMaterialTextures( const char* materialName ) const -> MaterialTextures;
    auto GetTexturesForLayers( const RgMeshPrimitiveInfo& primitive ) const -> std::array< MaterialTextures, 4 >;
    auto GetColorForLayers( const RgMeshPrimitiveInfo& primitive ) const -> std::array< RgColor4DPacked32, 4 >;

    // albedo of a material, EMPTY_TEXTURE_INDEX if not found
    uint32_t Find( const char* pTextureName ) const;

    uint32_t GetWaterNormalTextureIndex() const { return waterNormalTextureIndex; }
    uint32_t GetDirtMaskTextureIndex() const { return dirtMaskTextureIndex; }
    uint32_t GetSceneBuildingTextureIndex() const { return sceneBuildingTextureIndex; }

    // must be called once per frame, frees the textures that are no longer used by the GPU
    void OnFrameStart();
    // filter for textures with RG_SAMPLER_FILTER_AUTO
    void SetDynamicSamplerFilter( RgSamplerFilter filter );
    // SamplerManager::TryChangeMipLodBias (needs macOS 26, otherwise ignored)
    void SetMipLodBias( float bias );

    id< MTLBuffer > TableBuffer() const { return table; }
    // for RTGL1's GLSL shaders (METAL in ShaderCommonGLSLFunc.h): separate arrays of images and samplers
    id< MTLBuffer > ImageTableBuffer() const { return imageTable; }
    id< MTLBuffer > SamplerTableBuffer() const { return samplerTable; }
    // all textures, to make them resident
    const std::vector< id< MTLResource > >& LiveTextures() const;
    void            UseResources( id< MTLComputeCommandEncoder > enc ) const;
    void            UseResources( id< MTLRenderCommandEncoder > enc ) const;

private:
    struct Slot
    {
        id< MTLTexture >       texture{ nil };
        SamplerManager::Handle sampler{};
    };

    struct Material
    {
        MaterialTextures textures;
    };

    enum class ImportedType
    {
        ForStatic,
        ForReplacement,
    };

    uint32_t AllocSlot();
    void     FreeSlot( uint32_t slot );
    void     WriteTableEntry( uint32_t slot );

    id< MTLSamplerState > GetSampler( SamplerManager::Handle handle );

    // TextureManager::PrepareTexture
    uint32_t PrepareTexture( const std::optional< ImageLoader::ResultInfo >& info,
                             SamplerManager::Handle                          sampler,
                             bool                                            useMipmaps,
                             const char*                                     debugName,
                             std::optional< RgTextureSwizzling >             swizzling,
                             const std::filesystem::path&                    filepath );

    void MakeMaterial( std::string_view                                 materialName,
                       std::span< TextureOverrides >                    ovrd,
                       std::span< const SamplerManager::Handle >        samplers,
                       std::span< std::optional< RgTextureSwizzling > > swizzlings );

    bool     TryDestroyMaterial( const char* materialName );
    uint32_t CreateSingleTexture( const std::filesystem::path& path, bool isSRGB, const char* debugName );

    TextureOverrides::Loader OnlyKTX2LoaderIfNonDevMode();
    TextureOverrides::Loader AnyImageLoader();

private:
    id< MTLDevice >       device;
    id< MTLCommandQueue > queue;

    std::filesystem::path ovrdFolder;
    RgTextureSwizzling    pbrSwizzling;
    bool                  forceMinificationFilterLinear;
    bool                  forceNormalMapFilterLinear;
    RgSamplerFilter       dynamicFilter{ RG_SAMPLER_FILTER_LINEAR };
    float                 mipLodBias{ 0.0f };
    bool                  isdevmode{ false };

    std::unique_ptr< ImageLoader >    imageLoaderKtx;
    std::unique_ptr< ImageLoaderDev > imageLoaderRaw;

    // TableEntry per slot
    id< MTLBuffer > table;
    // MTLResourceID per slot
    id< MTLBuffer > imageTable;
    id< MTLBuffer > samplerTable;

    mutable std::vector< id< MTLResource > > liveTextures;
    mutable bool                             liveTexturesDirty{ true };

    std::vector< Slot >                                    slots;
    std::vector< uint32_t >                                freeSlots;
    rgl::string_map< Material >                            materials;
    rgl::string_map< ImportedType >                        importedMaterials;
    rgl::unordered_map< uint32_t, id< MTLSamplerState > > samplers;

    uint32_t waterNormalTextureIndex{ EMPTY_TEXTURE_INDEX };
    uint32_t dirtMaskTextureIndex{ EMPTY_TEXTURE_INDEX };
    uint32_t sceneBuildingTextureIndex{ EMPTY_TEXTURE_INDEX };

    struct PendingDelete
    {
        uint32_t slot;
        uint32_t framesLeft;
    };
    std::vector< PendingDelete > pendingDeletes;
};

}
