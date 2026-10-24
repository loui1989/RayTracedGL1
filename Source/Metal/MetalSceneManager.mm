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

// as in Scene.cpp, before any GLM include
#define GLM_FORCE_QUAT_DATA_XYZW 1

#include "MetalSceneManager.h"

#include "MetalLightManager.h"
#include "MetalScene.h"
#include "MetalTextureManager.h"

#include "Const.h"
#include "DrawFrameInfo.h"
#include "IFileDependency.h"
#include "SceneAnimation.h"
#include "TextureMeta.h"
#include "Utils.h"

#include <future>
#include <ranges>
#include <set>

namespace
{

// Scene.cpp
auto MakeGltfPath( const std::filesystem::path& base, std::string_view meshName ) -> std::filesystem::path
{
    auto exportName = std::string( meshName );

    std::ranges::replace( exportName, '\\', '_' );
    std::ranges::replace( exportName, '/', '_' );

    return base / exportName / ( exportName + ".gltf" );
}

auto GetGltfFilesSortedAlphabetically( const std::filesystem::path& folder ) -> std::set< std::filesystem::path >
{
    namespace fs = std::filesystem;

    if( folder.empty() || !exists( folder ) || !is_directory( folder ) )
    {
        return {};
    }

    auto gltfs = std::set< std::filesystem::path >{};
    try
    {
        for( const fs::directory_entry& entry : fs::directory_iterator{ folder } )
        {
            if( entry.is_regular_file() && RTGL1::MakeFileType( entry.path() ) == RTGL1::FileType::GLTF )
            {
                gltfs.insert( entry.path() );
            }
        }
    }
    catch( const std::filesystem::filesystem_error& e )
    {
        RTGL1::debug::Error( R"(directory_iterator failure: '{}'. path1: '{}'. path2: '{}')",
                             e.what(),
                             e.path1().string(),
                             e.path2().string() );
        return {};
    }
    return gltfs;
}

float OneIfNonZero( float v )
{
    return v < std::numeric_limits< float >::epsilon() ? 1.0f : v;
}

auto AddSuffix( const std::filesystem::path& base, std::string_view suffix ) -> std::filesystem::path
{
    auto p = base;
    p.replace_filename( base.stem().string() + std::string( suffix ) + base.extension().string() );
    return p;
}

}

RTGL1::MetalSceneManager::MetalSceneManager( const RgInstanceCreateInfo& info,
                                             const std::filesystem::path& ovrdFolder,
                                             MetalScene&                  _scene,
                                             MetalTextureManager&         _textures,
                                             MetalLightManager&           _lights,
                                             TextureMetaManager&          _textureMeta )
    : scene{ _scene }
    , textures{ _textures }
    , lights{ _lights }
    , textureMeta{ _textureMeta }
    , scenesFolder{ ovrdFolder / SCENES_FOLDER }
    , replacementsFolder{ ovrdFolder / REPLACEMENTS_FOLDER }
    , worldUp{ Utils::SafeNormalize( info.worldUp, { 0, 1, 0 } ) }
    , worldForward{ Utils::SafeNormalize( info.worldForward, { 0, 0, 1 } ) }
    , worldScale{ std::max( 0.0f, info.worldScale ) }
    , importedLightIntensityScaleDirectional{ OneIfNonZero( info.importedLightIntensityScaleDirectional ) }
    , importedLightIntensityScaleSphere{ OneIfNonZero( info.importedLightIntensityScaleSphere ) }
    , importedLightIntensityScaleSpot{ OneIfNonZero( info.importedLightIntensityScaleSpot ) }
{
}

auto RTGL1::MetalSceneManager::MakeImportExportParams() const -> ImportExportParams
{
    return ImportExportParams{
        .worldTransform =
            Utils::MakeTransform( Utils::Normalize( worldUp ), Utils::Normalize( worldForward ), worldScale ),
        .oneGameUnitInMeters                    = worldScale,
        .importedLightIntensityScaleDirectional = importedLightIntensityScaleDirectional,
        .importedLightIntensityScaleSphere      = importedLightIntensityScaleSphere,
        .importedLightIntensityScaleSpot        = importedLightIntensityScaleSpot,
    };
}

void RTGL1::MetalSceneManager::PrepareForFrame( uint32_t frameIndex, const RgStartFrameInfo& info )
{
    // Scene::PrepareForFrame
    ignoreExternalGeometry = !!info.ignoreExternalGeometry;
    dynamicUniqueIDs.clear();
    alreadyReplacedUniqueObjectIDs.clear();
    lastDynamicSun_uniqueId  = std::nullopt;
    staticSceneAnimationTime = info.staticSceneAnimationTime;

    // SceneImportExport::PrepareForFrame (without the export)
    {
        const auto mapName = std::string_view{ Utils::SafeCstr( info.pMapName ) };
        if( currentMap != mapName )
        {
            currentMap     = mapName;
            reimportStatic = true;
        }
    }

    // SceneImportExport::TryImportIfNew
    const bool newSceneRequested = reimportStatic;

    // An application uploads all its geometry (and provides all its textures) on a frame with
    // RG_STATIC_SCENE_STATUS_NEW_SCENE_STARTED, so the static scene gets the textures it waits for
    // (SHIPPING_HACK, see OnOriginalTextureProvided). But gzdoom draws no world on the first frame of a map,
    // so keep reporting the new scene until a frame that had world geometry.
    if( newSceneRequested )
    {
        newScenePending = true;
    }
    else if( newScenePending && worldUploadedInPrevFrame )
    {
        newScenePending = false;
    }
    worldUploadedInPrevFrame = false;
    if( reimportReplacements || reimportStatic )
    {
        // before importer, as it relies on texture properties
        textureMeta.RereadFromFiles( std::string_view{ currentMap } );

        NewScene( frameIndex, reimportReplacements );

        reimportReplacements = false;
        reimportStatic       = false;
    }

    if( info.pResultStaticSceneStatus )
    {
        *info.pResultStaticSceneStatus = 0;
        if( StaticSceneExists() )
        {
            *info.pResultStaticSceneStatus |= RG_STATIC_SCENE_STATUS_LOADED;
        }
        if( newScenePending )
        {
            *info.pResultStaticSceneStatus |= RG_STATIC_SCENE_STATUS_NEW_SCENE_STARTED;
        }
    }

    // SHIPPING_HACK: Scene::PrepareForFrame's Hack_PatchGeomInfoTransformForStatic
    for( const AnimatedObject& obj : animatedObjects )
    {
        auto mesh      = MakeMeshInfoFrom( obj.instancedName.c_str(), obj.model );
        mesh.transform = sceneanim::SampleAnimationObj( obj.model.animobj, obj.model.meshTransform, staticSceneAnimationTime );

        for( uint32_t i = 0; i < obj.model.primitives.size(); i++ )
        {
            MakeMeshPrimitiveInfoAndProcess( obj.model.primitives[ i ], i, [ & ]( const RgMeshPrimitiveInfo& prim ) {
                scene.AddReplacement( mesh, prim, textures );
            } );
        }
    }
}

void RTGL1::MetalSceneManager::SubmitStaticLights( uint32_t          frameIndex,
                                                   bool              isUnderwater,
                                                   RgColor4DPacked32 underwaterColor ) const
{
    for( const LightCopy& l : staticLights )
    {
        // SHIPPING_HACK begin - tint sun if underwater
        if( isUnderwater )
        {
            if( auto sun = std::get_if< RgLightDirectionalEXT >( &l.extension ) )
            {
                auto tintedSun  = RgLightDirectionalEXT{ *sun };
                tintedSun.color = underwaterColor;

                lights.Add( frameIndex,
                            LightCopy{
                                .base       = l.base,
                                .extension  = tintedSun,
                                .additional = l.additional,
                            } );
                continue;
            }
        }
        // SHIPPING_HACK end

        lights.Add( frameIndex, l );
    }
}

// Scene::NewScene
void RTGL1::MetalSceneManager::NewScene( uint32_t frameIndex, bool reimportReplacements )
{
    const auto params = MakeImportExportParams();

    staticUniqueIDs.clear();
    staticMeshNames.clear();
    staticLights.clear();
    cameraInfo_Imported     = {};
    cameraInfo_ImportedAnim = {};
    animatedObjects.clear();
    primitivesToUpdateTextures.clear();

    textures.FreeAllImportedMaterials( reimportReplacements );

    scene.BeginStatic( reimportReplacements );

    if( reimportReplacements )
    {
        replacements.clear();

        debug::Verbose( "Reading replacements..." );
        const auto gltfs = GetGltfFilesSortedAlphabetically( replacementsFolder );

        auto allImported = std::vector< std::future< std::unique_ptr< WholeModelFile > > >{};
        {
            // reverse alphabetical -- last ones have more priority
            for( const auto& p : std::ranges::reverse_view{ gltfs } )
            {
                allImported.push_back( std::async(
                    std::launch::async,
                    [ & ]( const std::filesystem::path& path ) -> std::unique_ptr< WholeModelFile > {
                        if( auto i = GltfImporter{ path, params, textureMeta, true } )
                        {
                            return std::make_unique< WholeModelFile >( i.Move() );
                        }
                        return {};
                    },
                    p ) );
            }
        }

        for( auto& ff : allImported )
        {
            auto wholeGltf = ff.valid() ? ff.get() : std::unique_ptr< WholeModelFile >{};
            if( !wholeGltf )
            {
                continue;
            }

            if( !wholeGltf->lights.empty() )
            {
                debug::Warning( "Ignoring non-attached lights from a replacements file" );
            }

            for( const auto& mat : wholeGltf->materials )
            {
                textures.TryCreateImportedMaterial(
                    mat.pTextureName, mat.fullPaths, mat.samplers, mat.pbrSwizzling, mat.isReplacement );
            }

            for( auto& [ meshName, meshSrc ] : wholeGltf->models )
            {
                auto [ iter, isNew ] = replacements.emplace( meshName, std::move( meshSrc ) );

                if( isNew )
                {
                    WholeModelFile::RawModelData& m = iter->second;

                    for( uint32_t index = 0; index < m.primitives.size(); index++ )
                    {
                        MakeMeshPrimitiveInfoAndProcess(
                            m.primitives[ index ], index, [ & ]( const RgMeshPrimitiveInfo& prim ) {
                                scene.CacheReplacement( meshName, prim, index );
                            } );

                        // save up some memory by not storing - as we uploaded already
                        m.primitives[ index ].vertices = {};
                        m.primitives[ index ].indices  = {};
                    }

                    if( m.primitives.empty() && m.localLights.empty() )
                    {
                        debug::Warning( "Replacement is empty, it doesn't have any primitives or lights: \'{}\'",
                                        meshName );
                    }
                }
                else
                {
                    debug::Warning( "Ignoring a replacement as it was already read from another .gltf file: \'{}\'",
                                    meshName );
                }
            }
        }
        debug::Verbose( "Replacements are ready" );
    }

    // SHIPPING_HACK begin
    auto trackTextureToReplace = rgl::string_set{};
    // SHIPPING_HACK end

    const auto staticSceneGltfPath = MakeGltfPath( scenesFolder, currentMap );

    if( auto staticScene = GltfImporter{ staticSceneGltfPath, params, textureMeta, false } )
    {
        WholeModelFile sceneFile = staticScene.Move();

        debug::Verbose( "Starting new static scene..." );

        if( auto patchScene =
                GltfImporter{ AddSuffix( staticSceneGltfPath, SCENE_PATCH_SUFFIX ), params, textureMeta, false } )
        {
            WholeModelFile patch = patchScene.Move();

            sceneFile.materials.insert( sceneFile.materials.end(),
                                        std::make_move_iterator( patch.materials.begin() ),
                                        std::make_move_iterator( patch.materials.end() ) );

            for( auto& [ name, mdl ] : patch.models )
            {
                auto f = sceneFile.models.find( name );
                if( f != sceneFile.models.end() )
                {
                    if( !Utils::AreAlmostSameTr( f->second.meshTransform, mdl.meshTransform ) )
                    {
                        debug::Warning( "Patch file contains node \'{}\' with one transform, but the base gltf "
                                        "file contains a node with same name which has ANOTHER transform. "
                                        "Expect incorrect patch file meshes. Base gltf file: {}",
                                        name,
                                        staticSceneGltfPath.string() );
                    }

                    f->second.primitives.insert( f->second.primitives.end(),
                                                 std::make_move_iterator( mdl.primitives.begin() ),
                                                 std::make_move_iterator( mdl.primitives.end() ) );
                }
                else
                {
                    sceneFile.models.emplace( std::move( name ), std::move( mdl ) );
                }
            }
        }

        for( const auto& mat : sceneFile.materials )
        {
            textures.TryCreateImportedMaterial(
                mat.pTextureName, mat.fullPaths, mat.samplers, mat.pbrSwizzling, mat.isReplacement );

            // SHIPPING_HACK begin
            if( mat.trackOriginalTexture && !mat.pTextureName.empty() )
            {
                trackTextureToReplace.insert( mat.pTextureName );
            }
            // SHIPPING_HACK end
        }

        for( auto& [ name, m ] : sceneFile.models )
        {
            const auto mesh = MakeMeshInfoFrom( name.c_str(), m );

            // SHIPPING_HACK: animated static objects are instanced each frame with a sampled transform
            const bool isAnimated = !IsAnimDataEmpty( m.animobj );
            auto       instancedName = std::string{ "\x01" "animated:" } + name;

            for( uint32_t i = 0; i < m.primitives.size(); i++ )
            {
                MakeMeshPrimitiveInfoAndProcess( m.primitives[ i ], i, [ & ]( const RgMeshPrimitiveInfo& prim ) {
                    if( isAnimated )
                    {
                        if( InsertPrimitiveInfo( PrimitiveUniqueID{ mesh, prim }, true, mesh, prim ) )
                        {
                            scene.CacheStaticInstanced( instancedName, prim, i );
                        }
                        return;
                    }

                    UploadResult ur = UploadPrimitive( frameIndex, mesh, prim, true );

                    // SHIPPING_HACK begin
                    if( ( ur == UploadResult::ExportableStatic || ur == UploadResult::Static ) &&
                        !Utils::IsCstrEmpty( prim.pTextureName ) && trackTextureToReplace.contains( prim.pTextureName ) )
                    {
                        primitivesToUpdateTextures[ prim.pTextureName ].push_back( PrimitiveUniqueID{ mesh, prim } );
                    }
                    // SHIPPING_HACK end
                } );
            }

            if( isAnimated )
            {
                auto copy = WholeModelFile::RawModelData{ m };
                for( auto& p : copy.primitives )
                {
                    p.vertices = {};
                    p.indices  = {};
                }
                animatedObjects.push_back( AnimatedObject{
                    .instancedName = std::move( instancedName ),
                    .model         = std::move( copy ),
                } );
            }

            if( !m.localLights.empty() )
            {
                debug::Warning( "Lights under the scene mesh ({}) are ignored, put them under the root node.", name );
            }
        }

        // camera
        if( sceneFile.camera )
        {
            cameraInfo_Imported = *( sceneFile.camera );
        }
        if( !IsAnimDataEmpty( sceneFile.animcamera ) )
        {
            cameraInfo_ImportedAnim = std::move( sceneFile.animcamera );
        }

        // global lights
        for( const auto& l : sceneFile.lights )
        {
            UploadLight( frameIndex, l, true );
        }

        if( sceneFile.lights.empty() )
        {
            debug::Warning( "Haven't found any lights in {}: Original exportable lights will be used",
                            staticScene.FilePath() );
        }
        debug::Verbose( "Static scene is ready" );
    }
    else
    {
        debug::Info( "New scene is empty" );
    }

    scene.SubmitStatic();
    debug::Info( "Static geometry was rebuilt" );
}

auto RTGL1::MetalSceneManager::UploadPrimitive( uint32_t                   frameIndex,
                                                const RgMeshInfo&          mesh,
                                                const RgMeshPrimitiveInfo& primitive ) -> UploadResult
{
    worldUploadedInPrevFrame = true;
    return UploadPrimitive( frameIndex, mesh, primitive, false );
}

// Scene::UploadPrimitive
auto RTGL1::MetalSceneManager::UploadPrimitive( uint32_t                   frameIndex,
                                                const RgMeshInfo&          mesh,
                                                const RgMeshPrimitiveInfo& primitive,
                                                bool                       isStatic ) -> UploadResult
{
    const auto uniqueID = PrimitiveUniqueID{ mesh, primitive };

    auto replacement = static_cast< const WholeModelFile::RawModelData* >( nullptr );

    if( !ignoreExternalGeometry )
    {
        if( !isStatic && mesh.isExportable && !Utils::IsCstrEmpty( mesh.pMeshName ) )
        {
            // if dynamic-exportable was already uploaded
            // (i.e. found a matching mesh inside a static scene)
            // otherwise, continue as dynamic
            if( StaticMeshExists( mesh ) )
            {
                return UploadResult::ExportableStatic;
            }

            if( mesh.flags & RG_MESH_EXPORT_AS_SEPARATE_FILE )
            {
                replacement = find_p( replacements, mesh.pMeshName );
            }
        }
    }

    if( !InsertPrimitiveInfo( uniqueID, isStatic, mesh, primitive ) )
    {
        return UploadResult::Fail;
    }

    if( !replacement )
    {
        const bool added = isStatic ? scene.AddStatic( mesh, primitive, uniqueID, textures )
                                    : scene.AddDynamic( mesh, primitive, uniqueID, textures );
        if( !added )
        {
            return UploadResult::Fail;
        }
    }
    else
    {
        assert( !isStatic );

        // multiple primitives can correspond to one mesh instance,
        // if a replacement for a mesh is present, upload it once
        if( !alreadyReplacedUniqueObjectIDs.contains( mesh.uniqueObjectID ) )
        {
            for( uint32_t i = 0; i < replacement->primitives.size(); i++ )
            {
                MakeMeshPrimitiveInfoAndProcess(
                    replacement->primitives[ i ], i, [ & ]( const RgMeshPrimitiveInfo& replacementPrim ) {
                        scene.AddReplacement( mesh, replacementPrim, textures );
                    } );
            }

            for( auto localLight : replacement->localLights )
            {
                assert( localLight.base.uniqueID != 0 && localLight.base.isExportable );

                auto hashCombine = []< typename T >( size_t seed, const T& v ) {
                    return seed ^ ( std::hash< T >{}( v ) + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 ) );
                };

                localLight.base.uniqueID     = hashCombine( localLight.base.uniqueID, mesh.uniqueObjectID );
                localLight.base.isExportable = false;

                if( localLight.additional &&
                    ( localLight.additional->flags & RG_LIGHT_ADDITIONAL_APPLY_PARENT_MESH_INTENSITY ) )
                {
                    std::visit( [ mult = mesh.localLightsIntensity ]( auto& ext ) { ext.intensity *= mult; },
                                localLight.extension );
                }

                UploadLight( frameIndex, localLight, false, &mesh.transform );
            }

            alreadyReplacedUniqueObjectIDs.insert( mesh.uniqueObjectID );
        }

        assert( mesh.isExportable );
        return UploadResult::ExportableDynamic;
    }

    return isStatic ? ( mesh.isExportable ? UploadResult::ExportableStatic : UploadResult::Static )
                    : ( mesh.isExportable ? UploadResult::ExportableDynamic : UploadResult::Dynamic );
}

bool RTGL1::MetalSceneManager::StaticMeshExists( const RgMeshInfo& mesh ) const
{
    if( !Utils::IsCstrEmpty( mesh.pMeshName ) )
    {
        // TODO: actually, need to consider RgMeshInfo::uniqueObjectID,
        // as there might be different instances of the same mesh
        if( staticMeshNames.contains( mesh.pMeshName ) )
        {
            return true;
        }
    }
    return false;
}

bool RTGL1::MetalSceneManager::StaticLightExists( const LightCopy& light ) const
{
    assert( light.base.isExportable );

    // TODO: compare ID-s?
    return !staticLights.empty();
}

bool RTGL1::MetalSceneManager::InsertPrimitiveInfo( const PrimitiveUniqueID&   uniqueID,
                                                    bool                       isStatic,
                                                    const RgMeshInfo&          mesh,
                                                    const RgMeshPrimitiveInfo& primitive )
{
    if( isStatic )
    {
        assert( !Utils::IsCstrEmpty( mesh.pMeshName ) );
        if( !Utils::IsCstrEmpty( mesh.pMeshName ) )
        {
            staticMeshNames.emplace( mesh.pMeshName );
        }

        if( !dynamicUniqueIDs.contains( uniqueID ) )
        {
            auto [ iter, isNew ] = staticUniqueIDs.emplace( uniqueID );
            if( isNew )
            {
                return true;
            }
        }
    }
    else
    {
        if( !staticUniqueIDs.contains( uniqueID ) )
        {
            auto [ iter, isNew ] = dynamicUniqueIDs.emplace( uniqueID );
            if( isNew )
            {
                return true;
            }
        }
    }

    debug::Warning( "Mesh primitive ({}) with ID ({}->{}): "
                    "Trying to upload but a primitive with the same ID already exists",
                    Utils::SafeCstr( mesh.pMeshName ),
                    mesh.uniqueObjectID,
                    primitive.primitiveIndexInMesh );
    return false;
}

bool RTGL1::MetalSceneManager::InsertLightInfo( bool isStatic, const LightCopy& light )
{
    if( isStatic )
    {
        // just check that there's no id collision
        auto foundSameId = std::ranges::find_if(
            staticLights, [ &light ]( const LightCopy& other ) { return other.base.uniqueID == light.base.uniqueID; } );

        if( foundSameId != staticLights.end() )
        {
            debug::Warning( "Trying add a static light with a uniqueID {} that other light already has",
                            light.base.uniqueID );
            return false;
        }

        // add to the list
        staticLights.push_back( light );
        return true;
    }
    return true;
}

// Scene::UploadLight
auto RTGL1::MetalSceneManager::UploadLight( uint32_t           frameIndex,
                                            const LightCopy&   light,
                                            bool               isStatic,
                                            const RgTransform* transform ) -> UploadResult
{
    assert( !isStatic || ( isStatic && !transform ) );

    const bool isExportable = light.base.isExportable;

    if( !isStatic )
    {
        if( isExportable )
        {
            if( StaticLightExists( light ) )
            {
                return UploadResult::ExportableStatic;
            }
        }
    }

    if( !InsertLightInfo( isStatic, light ) )
    {
        return UploadResult::Fail;
    }

    // adding static to light manager is done separately in SubmitStaticLights
    if( !isStatic )
    {
        lights.Add( frameIndex, light, transform );

        if( std::holds_alternative< RgLightDirectionalEXT >( light.extension ) )
        {
            lastDynamicSun_uniqueId = light.base.uniqueID;
        }
    }

    return isStatic ? ( isExportable ? UploadResult::ExportableStatic : UploadResult::Static )
                    : ( isExportable ? UploadResult::ExportableDynamic : UploadResult::Dynamic );
}

// Scene::GetCamera
auto RTGL1::MetalSceneManager::GetCamera( const std::optional< RgCameraInfo >& fromApplication,
                                          float                                fallbackAspect ) const
    -> std::optional< RgCameraInfo >
{
    if( cameraInfo_Imported && fromApplication )
    {
        auto modified = RgCameraInfo{ *fromApplication };
        {
            const auto imp =
                sceneanim::SampleAnimation( cameraInfo_ImportedAnim, *cameraInfo_Imported, staticSceneAnimationTime );

            modified.fovYRadians = imp.fovYRadians;
            modified.position    = imp.position;
            modified.up          = imp.up;
            modified.right       = imp.right;
            // the view is made from the position and the orientation above
            modified.pView = nullptr;
        }
        return modified;
    }
    if( fromApplication )
    {
        return fromApplication;
    }
    if( cameraInfo_Imported )
    {
        auto modified =
            sceneanim::SampleAnimation( cameraInfo_ImportedAnim, *cameraInfo_Imported, staticSceneAnimationTime );
        modified.aspect = fallbackAspect;
        modified.pView  = nullptr;
        modified.pNext  = nullptr;
        return modified;
    }
    return std::nullopt;
}

void RTGL1::MetalSceneManager::OnOriginalTextureProvided( const char* pTextureName )
{
    if( Utils::IsCstrEmpty( pTextureName ) )
    {
        return;
    }
    if( auto found = primitivesToUpdateTextures.find( pTextureName ); found != primitivesToUpdateTextures.end() )
    {
        const MaterialTextures mt = textures.GetMaterialTextures( pTextureName );
        for( const PrimitiveUniqueID& id : found->second )
        {
            scene.PatchStaticTextures( id, mt );
        }
    }
}

auto RTGL1::MetalSceneManager::TryGetVolumetricLight( const RgFloat3D& cameraPos ) const -> std::optional< uint64_t >
{
    return lights.TryGetVolumetricLight( cameraPos, staticLights, lastDynamicSun_uniqueId );
}
