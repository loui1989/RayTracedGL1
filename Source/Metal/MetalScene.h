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
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

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

    // Thief RT (0028): whether the next AddDynamic's primitive may be repeated (RepeatDynamic):
    // nothing else came of its upload (no light attached to it, no rasterized copy)
    void SetNextRepeatable( bool r ) { nextRepeatable = r; }
    // Thief RT (0028): a primitive given before with RT_PRIMITIVE_UNCHANGED (kept in the arena),
    // again as it was last given, without its data. False if it can't be (not kept, its record
    // gone): then it must be given whole.
    bool RepeatDynamic( const PrimitiveUniqueID& uniqueID, const MetalTextureManager& textures );

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

    // Debug: the size of the last built scene, for RTGL1_METAL_PROFILE
    struct Stats
    {
        size_t dynamicGeometries, dynamicTriangles, dynamicVertices, staticGeometries, tlasInstances, geometryInfos;
        size_t polishedGeometries;
        // Raze RT: the kept (stable) part of the dynamic geometry, and its rebuilds since the last stats
        size_t stableGeometries, stableTriangles, stableRebuilds, stableCells;
        // models instanced with their kept BLAS (modelBlas), the rest rebuilt each frame
        size_t instancedModels, modelBlases, modelBuilds, volatileGeometries, volatileTriangles;
        // Thief RT: the frames' dynamic BLAS-es refitted and built since the last stats
        size_t dynamicRefits, dynamicBuilds;
        size_t movingObjects;   // with a BLAS of their own (objectBlas)
        size_t objectsKept;     // of them, as they were (their BLAS used again as it was), this frame
        size_t objectKeeps, objectPolishes;   // since the start
        // Thief RT (0028): of the dynamic geometry, kept in the arena (not copied this frame),
        // repeated without its data, and copied into the frame's buffers this frame
        size_t retainedGeometries, retainedVertices, repeatedGeometries, retainedCopies;
        size_t arenaVertices, arenaCapacity;   // in use, and room for (vertices)
        // Thief RT (0032): the tracing test in force (rgThiefTracingTest), and the TLAS instances
        // by kind: kept pieces of the map, moving objects' own, and those with see-through
        // (alpha tested) triangles, with how many triangles
        int    tracingTest;
        size_t pieceInstances, objectInstances, alphaInstances, alphaTriangles;
    };
    Stats lastStats{};

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
        // Doom 64 RT: RG_MESH_PRIMITIVE_SHADOW_ONLY, seen only by shadow rays (INSTANCE_MASK_RESERVED_0)
        ShadowOnly,
    };
    enum class PassThrough : uint8_t
    {
        Opaque,
        AlphaTested,
        Refract,
        // Raze RT: opaque, and seen only from the front (RT_PRIMITIVE_ONE_SIDED: a Build wall
        // or floor): the only instances with triangle culling, the others disable it
        OpaqueOneSided,
    };
    static constexpr uint32_t PassThroughCount = 4;
    struct Group
    {
        Visibility  visibility;
        PassThrough passThrough;

        bool     operator==( const Group& ) const = default;
        uint32_t ToIndex() const { return uint32_t( visibility ) * PassThroughCount + uint32_t( passThrough ); }
    };
    static constexpr uint32_t GroupCount = 6 * PassThroughCount;

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
        // Raze RT: what the BLAS sees of it (positions, indices, transform), for the stable geometry
        uint64_t           contentKey{ 0 };
        uint32_t           triangles{ 0 };
        // and which piece of the map it is in (see StableGroup)
        int64_t            cell{ 0 };
        // its shape alone (0: not transformed), see modelBlas
        uint64_t           localKey{ 0 };
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
        // Thief RT: what it holds (its geometries' ids, triangle counts, kind), when
        // built refittable (0: not), and how many refits since it was last built
        uint64_t refitSig{ 0 };
        uint32_t refits{ 0 };
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
    void BuildBlas( id< MTLAccelerationStructureCommandEncoder > enc,
                    BuiltBlas&                                   dst,
                    const std::vector< const Geometry* >&        geoms,
                    id< MTLBuffer >                              vertices,
                    id< MTLBuffer >                              indices,
                    bool                                         bakeTransforms,
                    bool                                         fastBuild,
                    NSString*                                    label,
                    const BuiltBlas*                             refitFrom = nullptr );

    // Raze RT: stable geometry. A game that sends all of its world as dynamic geometry every
    // frame (Raze: a Build map's walls and floors) had all of it rebuilt into a BLAS each frame,
    // built for speed of building, not of tracing. The geometries that come again unchanged
    // (by contentKey) for a few frames go into a BLAS of their own, built for tracing and kept;
    // it is used for as long as all of its geometries come again. Its geometry infos are this
    // frame's (the shaders read the vertices of this frame's dynamic buffers, the same ones).
    // A kept BLAS a piece of the map (a cell of StableCellSize metres): one for all of it was
    // rebuilt whole whenever any of its geometries changed or was no longer sent -- a door, a
    // room left out of reach as the player walks: a few times a second, 150k+ triangles each.
    // RTGL1_METAL_NO_STABLE=1 turns it off.
    struct StableGroup
    {
        std::vector< uint64_t > keys; // of its geometries, in the BLAS order
        BuiltBlas               blas;
        size_t                  triangles{ 0 };
        uint32_t                builtFrame{ 0 };
    };
    void RetireStable( StableGroup& sg );

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

    // Thief RT: a primitive the application marks unchanged since it last gave it
    // (RT_PRIMITIVE_UNCHANGED, 1u << 29): its content key and cell are as they were then,
    // not found again from its every vertex each frame
    //
    // Thief RT (0028): and while it comes unchanged, its vertices, indices and texture
    // coordinates are kept in the arena -- the first part of each frame's dynamic buffers,
    // the same place in all of them: written into a frame's buffers once, not copied every
    // frame (the frame's other dynamic data follows the arena). Given as it was last given
    // (RepeatDynamic), it's its Geometry again, and nothing of it is copied at all.
    enum ArenaKind : uint32_t
    {
        ArenaVertices,
        ArenaIndices,
        ArenaLayer1,
        ArenaLayer2,
        ArenaLayer3,
        ArenaKindCount,
    };
    struct KeptKeys
    {
        uint64_t    contentKey{ 0 };
        uint64_t    localKey{ 0 };
        int64_t     cell{ 0 };
        uint32_t    vertexCount{ 0 };
        uint32_t    indexCount{ 0 };
        uint32_t    frame{ 0 };
        RgTransform transform{};

        // in the arena: where (elements), how much, which layers, and in which frames'
        // buffers it's been written (a bit for each of the ring)
        bool     retained{ false };
        uint8_t  written{ 0 };
        uint8_t  layers{ 0 };
        uint32_t at[ ArenaKindCount ]{};
        uint32_t retainedVertices{ 0 };
        uint32_t retainedIndices{ 0 };
        // as last given, for RepeatDynamic: its geometry and group, and the names its
        // textures were found by (found again when the materials have changed since)
        std::optional< Geometry > repeat;
        uint32_t                  groupIndex{ 0 };
        uint64_t                  materialsGeneration{ 0 };
        std::string               textureNames[ 4 ];
    };
    struct ArenaAllocator
    {
        uint32_t                       top{ 0 };
        uint32_t                       used{ 0 };
        std::map< uint32_t, uint32_t > freeRanges; // where -> how many
        bool                           Alloc( uint32_t count, uint32_t limit, uint32_t* at );
        void                           Free( uint32_t at, uint32_t count );
    };
    bool         TryRetain( KeptKeys& k, uint32_t vertexCount, uint32_t indexCount, uint8_t layers );
    void         ReleaseRetained( KeptKeys& k );
    void         WriteRetained( FrameData& f, const KeptKeys& k, const RgMeshPrimitiveInfo& prim );
    void         CopyRetained( const FrameData& from, FrameData& to, const KeptKeys& k );
    VertexRange  RetainedRange( const KeptKeys& k ) const;
    id< MTLBuffer > __strong& ArenaBuffer( FrameData& f, uint32_t kind );
    static size_t ArenaElementSize( uint32_t kind );
    void         EnsureArena( FrameData& f );
    std::array< ArenaAllocator, ArenaKindCount > arena;
    // this frame's room in the arena (elements); the frame's other dynamic data is after it
    std::array< uint32_t, ArenaKindCount >       arenaCapacity{};
    // what didn't fit this frame: room made for it at the next frame's start
    std::array< uint64_t, ArenaKindCount >       arenaShortfall{};
    bool                                         nextRepeatable{ false };
    size_t                                       retainedCount{ 0 }, retainedVertexCount{ 0 }, retainedIndexCount{ 0 };
    size_t                                       repeatedCount{ 0 }, retainedCopyCount{ 0 };
    std::unordered_map< PrimitiveUniqueID, KeptKeys > keptKeys;
    uint32_t                                          keptFrame{ 0 };

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
        // to detect a texture change (e.g. a sprite's animation frame), see GEOM_INST_FLAGS2_NO_HISTORY
        uint32_t texture_base;
        uint32_t texture_base_E;
    };
    rgl::unordered_map< PrimitiveUniqueID, PrevInfo > prevInfos;
    uint32_t                                          prevGeomCount{ 0 };

    // Raze RT: stable geometry (see StableGroup)
    std::array< std::unordered_map< int64_t, StableGroup >, GroupCount > stable;
    std::unordered_map< uint64_t, uint32_t > keyAge; // a content key -> frames in a row it came
    uint32_t                                 frameCounter{ 0 };
    size_t                                   stableRebuilds{ 0 };
    // Thief RT: dynamic BLAS refits and builds (see BuildBlas's refitFrom)
    size_t                                   dynamicRefits{ 0 };
    size_t                                   dynamicBuilds{ 0 };
    // Raze RT: a model that moves but keeps its shape (a spinning voxel pickup, a swinging
    // door's voxel) changed its content key every frame: rebuilt with the dynamic BLAS each
    // frame. By its shape alone (Geometry::localKey): a BLAS in its own space, kept while it
    // is seen, a TLAS instance with its transform for each one of it
    struct ModelBlas
    {
        BuiltBlas blas;
        uint32_t  lastFrame{ 0 };
    };
    std::array< std::unordered_map< uint64_t, ModelBlas >, GroupCount > modelBlas;
    // Thief RT: each moving object of some size (a guard, a swinging door) its own BLAS,
    // one for each frame of the ring, each refitted from the one of the frame before
    // (BuildBlas's refitFrom): the frame's one shared dynamic BLAS was built anew whenever
    // anything in it changed -- a guard coming into view, a particle -- and with it every
    // other moving thing. Its key: the game's object (Thief RT gives an object's parts the
    // ids 0x500000000 + object << 12 + part, so objectId >> 12).
    //
    // An object that comes the same as the frame before (a statue, a crate, a tree kept out of
    // view for its shadow -- most of them, most frames) keeps the BLAS it had: nothing is
    // refitted or built for it. So the BLAS a frame uses is the slot last written (useSlot),
    // not the frame's ring's; a changed one is written into a slot no frame in flight uses
    // (usedFrame), refitted from the one before. Unchanged a while after a refit or a fast
    // build, it is built once more for tracing. (RTGL1_METAL_NO_OBJECT_KEEP=1: refitted each
    // frame, as before.)
    struct ObjectBlas
    {
        std::array< BuiltBlas, RingSize > slots;
        std::array< uint32_t, RingSize >  usedFrame{};   // 0: never
        uint32_t                          useSlot{ ~0u };
        uint32_t                          lastFrame{ 0 };
        uint64_t                          contentSig{ 0 };
        uint32_t                          sameFrames{ 0 };
        bool                              polished{ false };
    };
    size_t objectKeeps{ 0 }, objectPolishes{ 0 };
    std::array< std::unordered_map< uint64_t, ObjectBlas >, GroupCount > objectBlas;
    std::unordered_map< uint64_t, uint32_t >                             localAge;
    size_t                                                               modelBuilds{ 0 };
};

}
