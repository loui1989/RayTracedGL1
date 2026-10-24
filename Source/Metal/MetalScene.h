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
#include "UniqueID.h"

#include "Material.h"

#include "Generated/ShaderCommonC.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace RTGL1
{

class MetalBindingTable;
class MetalTextureManager;

// Ray traced geometry: the same data as ASManager + GeomInfoManager, in Metal acceleration structures.
//
// - Static geometry (an imported scene) and replacements are uploaded once, to the "static" buffers.
// - Dynamic geometry is uploaded every frame, to the "dynamic" buffers of the frame.
// - Primitives are grouped (as VertexCollectorFilterType: visibility x pass-through type),
//   a group is a primitive acceleration structure, each primitive is its geometry.
//   Static and dynamic groups have their mesh transforms baked into the BLAS,
//   a replacement's group is instanced in the TLAS with the mesh transform.
// - A TLAS instance's user ID is ( index of its first ShGeometryInstance << 4 ) | INSTANCE_CUSTOM_INDEX_FLAG_*,
//   see RAY_QUERY_GROUPED_INSTANCES in RaygenCommon.h.
// - ShGeometryInstance-s are matched with the previous frame's by PrimitiveUniqueID, for motion vectors.
class MetalScene
{
public:
    explicit MetalScene( id< MTLDevice > device, id< MTLCommandQueue > queue );
    ~MetalScene();

    MetalScene( const MetalScene& other )            = delete;
    MetalScene& operator=( const MetalScene& other ) = delete;

    void BeginFrame( uint32_t frameIndex );

    // ASManager::AddMeshPrimitive for dynamic geometry
    bool AddDynamic( const RgMeshInfo&          mesh,
                     const RgMeshPrimitiveInfo& prim,
                     const PrimitiveUniqueID&   uniqueID,
                     const MetalTextureManager& textures );

    // Static geometry: between BeginStatic and SubmitStatic.
    // If 'withReplacements', the replacements are also reset, and must be cached again.
    void BeginStatic( bool withReplacements );
    bool AddStatic( const RgMeshInfo&          mesh,
                    const RgMeshPrimitiveInfo& prim,
                    const PrimitiveUniqueID&   uniqueID,
                    const MetalTextureManager& textures );
    // ASManager::CacheReplacement: the vertex data of a replacement's primitive,
    // primitives must be cached in the order of their indices
    void CacheReplacement( std::string_view meshName, const RgMeshPrimitiveInfo& prim, uint32_t index );
    // A static object that is instanced with a transform each frame (an animated object of an imported scene),
    // its primitives are added each frame with AddReplacement and mesh.pMeshName = 'name'
    void CacheStaticInstanced( std::string_view name, const RgMeshPrimitiveInfo& prim, uint32_t index );
    // uploads the static vertex data and builds the static acceleration structures
    void SubmitStatic();

    // SHIPPING_HACK: ASManager::Hack_PatchTexturesForStaticPrimitive
    void PatchStaticTextures( const PrimitiveUniqueID& uniqueID, const MaterialTextures& textures );

    // A replacement's primitive, as an instance of the cached replacement
    bool AddReplacement( const RgMeshInfo&          mesh,
                         const RgMeshPrimitiveInfo& replacementPrim,
                         const MetalTextureManager& textures );

    // Builds the dynamic acceleration structures and the TLAS, uploads the geometry infos.
    // Returns false, if there's nothing to trace.
    bool Build( id< MTLCommandBuffer > cmd, uint32_t rayCullMaskWorld );

    // VertexBufferStatic, VertexBufferDynamic, ..., GeometryInstances, GeometryIndicesPrevToCur, topLevelAS
    void Bind( MetalBindingTable& table ) const;

private:
    // VertexCollectorFilterType: primary visibility x pass-through type
    enum class Visibility : uint8_t
    {
        World0,
        World1,
        World2,
        FirstPerson,
        FirstPersonViewer,
    };
    enum class PassThrough : uint8_t
    {
        Opaque,
        AlphaTested,
        Refract,
    };
    struct Group
    {
        Visibility  visibility;
        PassThrough passThrough;

        bool     operator==( const Group& ) const = default;
        uint32_t ToIndex() const { return uint32_t( visibility ) * 3 + uint32_t( passThrough ); }
    };
    static constexpr uint32_t GroupCount = 5 * 3;

    static Group MakeGroup( const RgMeshInfo& mesh, const RgMeshPrimitiveInfo& prim );

    // vertex data of primitives
    struct VertexStorage
    {
        std::vector< ShVertex >  vertices;
        std::vector< uint32_t >  indices;
        std::vector< RgFloat2D > texCoordLayers[ 3 ];

        void Clear();
    };

    struct VertexRange
    {
        uint32_t firstVertex;
        uint32_t firstIndex;
        uint32_t vertexCount;
        uint32_t indexCount;
        uint32_t firstVertexLayer[ 3 ];
    };
    static std::optional< VertexRange > Store( VertexStorage& dst, const RgMeshPrimitiveInfo& prim );
    static VertexRange                  Offset( const VertexRange& r, const VertexRange& base );


    // a geometry of a BLAS
    struct Geometry
    {
        ShGeometryInstance info;
        PrimitiveUniqueID  uniqueID;
        bool               noMotionVectors;
        bool               alphaTested;
        RgTransform        bakedTransform;
    };
    static ShGeometryInstance MakeGeomInfo( const RgMeshInfo&          mesh,
                                            const RgMeshPrimitiveInfo& prim,
                                            const VertexRange&         range,
                                            bool                       isDynamic,
                                            const MetalTextureManager& textures );

    struct BuiltBlas
    {
        id< MTLAccelerationStructure > blas{ nil };
        id< MTLBuffer >                scratch{ nil };
        id< MTLBuffer >                transforms{ nil };
    };

    // geometries of a group, in the BLAS order
    using GroupedGeometries = std::array< std::vector< Geometry >, GroupCount >;

    // a mesh which BLAS-es are in its local space, instanced in the TLAS with a transform
    struct Replacement
    {
        struct Primitive
        {
            VertexRange range;
            Group       group;
            bool        alphaTested;
            // index of the geometry in its group's BLAS
            uint32_t    geometryIndex;
        };
        std::vector< Primitive > primitives;
        // per group
        std::array< BuiltBlas, GroupCount > blas;
    };

    static void CacheInstanced( Replacement& dst, VertexStorage& storage, const RgMeshPrimitiveInfo& prim, uint32_t index );
    void        BuildInstancedBlas( id< MTLAccelerationStructureCommandEncoder > enc,
                                    Replacement&                                 r,
                                    const VertexRange&                           base );

    // an instance of a replacement's group, gathered during a frame
    struct ReplacementInstance
    {
        const Replacement*                 replacement;
        // offset of the replacement's data in the static buffers
        VertexRange                        base;
        Group                              group;
        RgTransform                        transform;
        RgMeshInfoFlags                    meshFlags;
        uint64_t                           uniqueObjectID;
        // by the geometry index in the group's BLAS
        std::vector< std::optional< Geometry > > geometries;
    };

    struct FrameData
    {
        id< MTLBuffer > vertices;
        id< MTLBuffer > indices;
        id< MTLBuffer > texCoordLayers[ 3 ];
        id< MTLBuffer > geometries;
        id< MTLBuffer > matchPrev;
        id< MTLBuffer > instances;
        id< MTLBuffer > scratchTlas;

        std::array< BuiltBlas, GroupCount > dynamicBlas;
        id< MTLAccelerationStructure >      tlas;

        // all BLAS-es that the TLAS references
        std::vector< id< MTLAccelerationStructure > > usedBlas;
    };

    // a ring of MAX_FRAMES_IN_FLIGHT + 1: while the CPU writes the current frame's data,
    // a frame in flight might still read the previous frame's data as "Prev"
    static constexpr uint32_t RingSize = MAX_FRAMES_IN_FLIGHT + 1;

    const FrameData& Cur() const { return frames[ ring ]; }
    FrameData&       Cur() { return frames[ ring ]; }
    const FrameData& Prev() const { return frames[ ( ring + RingSize - 1 ) % RingSize ]; }

    void BuildBlas( id< MTLAccelerationStructureCommandEncoder > enc,
                    BuiltBlas&                                   dst,
                    const std::vector< Geometry >&               geoms,
                    id< MTLBuffer >                              vertices,
                    id< MTLBuffer >                              indices,
                    bool                                         bakeTransforms,
                    bool                                         fastBuild,
                    NSString*                                    label );

private:
    id< MTLDevice >       device;
    id< MTLCommandQueue > queue;

    FrameData frames[ RingSize ];
    uint32_t  ring{ 0 };

    // replaced static resources, kept alive while the frames in flight might use them
    struct Retired
    {
        std::vector< id > resources;
        uint32_t          framesLeft;
    };
    std::vector< Retired > retired;

    // dynamic, gathered on CPU during a frame
    VertexStorage                      dynamicStorage;
    GroupedGeometries                  dynamicGeoms;
    std::vector< ReplacementInstance > replacementInstances;
    size_t                             frameGeomCount{ 0 };

    // static: replacements, then the static scene, in the same buffers;
    // static scene's vertex ranges are relative to the end of replacements' data (staticBase)
    VertexStorage                       replacementStorage;
    VertexStorage                       staticStorage;
    rgl::string_map< Replacement >      replacements;
    rgl::string_map< Replacement >      staticInstanced;
    VertexRange                         staticBase{};
    GroupedGeometries                   staticGeoms;
    std::array< BuiltBlas, GroupCount > staticBlas;
    id< MTLBuffer >                     staticVertices{ nil };
    id< MTLBuffer >                     staticIndices{ nil };
    id< MTLBuffer >                     staticTexCoordLayers[ 3 ]{};
    bool                                makingStatic{ false };
    bool                                replacementsChanged{ false };

    // GeomInfoManager: the previous frame's data, by unique ID
    struct PrevInfo
    {
        uint32_t baseVertexIndex;
        uint32_t baseIndexIndex;
        uint32_t vertexCount;
        uint32_t indexCount;
        float    model_0[ 4 ];
        float    model_1[ 4 ];
        float    model_2[ 4 ];
        // index of ShGeometryInstance in the previous frame
        uint32_t globalIndex;
        // to detect a texture change (e.g. a sprite's animation frame), see GEOM_INST_FLAG_NO_HISTORY
        uint32_t texture_base;
        uint32_t texture_base_E;
    };
    rgl::unordered_map< PrimitiveUniqueID, PrevInfo > prevInfos;
    uint32_t                                          prevGeomCount{ 0 };
};

}
