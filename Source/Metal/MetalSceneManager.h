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

#include "Common.h"
#include "Containers.h"
#include "GltfImporter.h"
#include "UniqueID.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace RTGL1
{

class MetalScene;
class MetalTextureManager;
class MetalLightManager;
class TextureMetaManager;

// The same as Scene + SceneImportExport (without the export):
// static scenes imported from <ovrd>/scenes/<map>/<map>.gltf, mesh replacements from <ovrd>/replace,
// static lights, and the mapping of the uploaded primitives to static / dynamic / replacement geometry.
class MetalSceneManager
{
public:
    enum class UploadResult
    {
        Fail,
        Static,
        Dynamic,
        ExportableStatic,
        ExportableDynamic,
    };

    MetalSceneManager( const RgInstanceCreateInfo& info,
                       const std::filesystem::path& ovrdFolder,
                       MetalScene&                  scene,
                       MetalTextureManager&         textures,
                       MetalLightManager&           lights,
                       TextureMetaManager&          textureMeta );

    MetalSceneManager( const MetalSceneManager& other )            = delete;
    MetalSceneManager& operator=( const MetalSceneManager& other ) = delete;

    // Scene::PrepareForFrame, SceneImportExport::PrepareForFrame + TryImportIfNew
    void PrepareForFrame( uint32_t frameIndex, const RgStartFrameInfo& info );
    void SubmitStaticLights( uint32_t frameIndex, bool isUnderwater, RgColor4DPacked32 underwaterColor ) const;

    UploadResult UploadPrimitive( uint32_t frameIndex, const RgMeshInfo& mesh, const RgMeshPrimitiveInfo& primitive );
    UploadResult UploadLight( uint32_t           frameIndex,
                              const LightCopy&   light,
                              bool               isStatic,
                              const RgTransform* transform = nullptr );

    // Scene::GetCamera: the imported scene's camera might override the application's
    std::optional< RgCameraInfo > GetCamera( const std::optional< RgCameraInfo >& fromApplication,
                                             float                                fallbackAspect ) const;

    // Scene::TryGetVolumetricLight
    auto TryGetVolumetricLight( const RgFloat3D& cameraPos ) const -> std::optional< uint64_t >;

    // SHIPPING_HACK: a texture was provided, update the static geometry that waits for it
    void OnOriginalTextureProvided( const char* pTextureName );

    std::string_view GetImportMapName() const { return currentMap; }
    float            GetWorldScale() const { return worldScale; }
    bool             StaticSceneExists() const { return !staticMeshNames.empty(); }

private:
    void         NewScene( uint32_t frameIndex, bool reimportReplacements );
    UploadResult UploadPrimitive( uint32_t                   frameIndex,
                                  const RgMeshInfo&          mesh,
                                  const RgMeshPrimitiveInfo& primitive,
                                  bool                       isStatic );
    auto MakeImportExportParams() const -> ImportExportParams;

    bool StaticMeshExists( const RgMeshInfo& mesh ) const;
    bool StaticLightExists( const LightCopy& light ) const;
    bool InsertPrimitiveInfo( const PrimitiveUniqueID&   uniqueID,
                              bool                       isStatic,
                              const RgMeshInfo&          mesh,
                              const RgMeshPrimitiveInfo& primitive );
    bool InsertLightInfo( bool isStatic, const LightCopy& light );

private:
    MetalScene&          scene;
    MetalTextureManager& textures;
    MetalLightManager&   lights;
    TextureMetaManager&  textureMeta;

    // SceneImportExport
    std::filesystem::path scenesFolder;
    std::filesystem::path replacementsFolder;
    std::string           currentMap{};
    bool                  reimportStatic{ false };
    bool                  reimportReplacements{ true };
    bool                  newScenePending{ false };
    bool                  worldUploadedInPrevFrame{ false };
    RgFloat3D             worldUp;
    RgFloat3D             worldForward;
    float                 worldScale;
    float                 importedLightIntensityScaleDirectional;
    float                 importedLightIntensityScaleSphere;
    float                 importedLightIntensityScaleSpot;

    // Scene
    rgl::unordered_set< PrimitiveUniqueID > dynamicUniqueIDs;
    rgl::unordered_set< uint64_t >          alreadyReplacedUniqueObjectIDs;
    rgl::unordered_set< PrimitiveUniqueID > staticUniqueIDs;
    rgl::string_set                         staticMeshNames;
    std::vector< LightCopy >                staticLights;
    std::optional< uint64_t >               lastDynamicSun_uniqueId{};

    std::optional< RgCameraInfo > cameraInfo_Imported{};
    AnimationData                 cameraInfo_ImportedAnim{};
    float                         staticSceneAnimationTime{ 0 };

    rgl::string_map< WholeModelFile::RawModelData > replacements;

    bool ignoreExternalGeometry{ false };

    // static objects with animations: instanced each frame (see MetalScene::CacheStaticInstanced)
    struct AnimatedObject
    {
        std::string                  instancedName;
        WholeModelFile::RawModelData model;
    };
    std::vector< AnimatedObject > animatedObjects;

    // SHIPPING_HACK
    rgl::string_map< std::vector< PrimitiveUniqueID > > primitivesToUpdateTextures;
};

}
