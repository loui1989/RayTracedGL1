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


#if defined(VERTEX_PREPROCESS_PARTIAL_DYNAMIC)

    #define GET_POSITIONS getDynamicVerticesPositions
    #define GET_NORMALS getDynamicVerticesNormals
    #define SET_NORMALS setDynamicVerticesNormals
    #define INDICES dynamicIndices

#elif defined(VERTEX_PREPROCESS_PARTIAL_STATIC_ALL) || defined(VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE)

    #define GET_POSITIONS getStaticVerticesPositions
    #define GET_NORMALS getStaticVerticesNormals
    #define SET_NORMALS setStaticVerticesNormals
    #define INDICES staticIndices

#else
    #error
#endif




// translate from local to global geom index
const int geomIndexOffset = globalUniform.instanceGeomInfoOffset[tlasInstanceIndex / 4][tlasInstanceIndex % 4];
const int geomCount = globalUniform.instanceGeomCount[tlasInstanceIndex / 4][tlasInstanceIndex % 4];

// Flat normals for the geometries that need them. A geometry's triangles share vertices (models),
// and a vertex takes the normal of its last triangle, so the triangles of one geometry are
// processed in order by one thread, and the geometries are spread over the group's threads.
// Only very large geometries (big water / brush batches, which would keep one thread busy
// for long) are split between all threads. Degenerate triangles don't write a normal (NaN,
// which made the model black).
#define VERTPREPROC_PARALLEL_MIN_TRIANGLES 2048

#define VERTPREPROC_TRIANGLE( inst, tri )                                                           \
    {                                                                                               \
        uvec3 vertexIndices;                                                                        \
        if( useIndices )                                                                            \
        {                                                                                           \
            const uint i  = inst.baseIndexIndex + ( tri ) * 3;                                      \
            vertexIndices = uvec3( inst.baseVertexIndex + INDICES[ i + 0 ],                         \
                                   inst.baseVertexIndex + INDICES[ i + 1 ],                         \
                                   inst.baseVertexIndex + INDICES[ i + 2 ] );                       \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            const uint v  = inst.baseVertexIndex + ( tri ) * 3;                                     \
            vertexIndices = uvec3( v + 0, v + 1, v + 2 );                                           \
        }                                                                                           \
        const vec3 p0 = GET_POSITIONS( vertexIndices[ 0 ] );                                        \
        const vec3 p1 = GET_POSITIONS( vertexIndices[ 1 ] );                                        \
        const vec3 p2 = GET_POSITIONS( vertexIndices[ 2 ] );                                        \
        const vec3 c  = cross( p1 - p0, p2 - p0 );                                                  \
        if( dot( c, c ) > 1e-12 )                                                                   \
        {                                                                                           \
            const vec3 localNormal = normalSign * normalize( c );                                   \
            SET_NORMALS( vertexIndices[ 0 ], localNormal );                                         \
            SET_NORMALS( vertexIndices[ 1 ], localNormal );                                         \
            SET_NORMALS( vertexIndices[ 2 ], localNormal );                                         \
        }                                                                                           \
    }

// pass 1: the geometries spread over the threads, each processed in order by one thread
for (uint localGeomIndex = gl_LocalInvocationID.x; localGeomIndex < geomCount; localGeomIndex += gl_WorkGroupSize.x)
{
    const ShGeometryInstance inst = geometryInstances[geomIndexOffset + localGeomIndex];

#if defined(VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE)
    if ((inst.flags & GEOM_INST_FLAG_IS_MOVABLE) == 0)
    {
        continue;
    }
#endif
    if ((inst.flags & GEOM_INST_FLAG_GENERATE_NORMALS) == 0)
    {
        continue;
    }

    const bool  useIndices = inst.baseIndexIndex != UINT32_MAX;
    const float normalSign = float((inst.flags & GEOM_INST_FLAG_INVERTED_NORMALS) == 0) * 2.0 - 1.0;
    const uint  triCount   = useIndices ? inst.indexCount / 3 : inst.vertexCount / 3;

    if (triCount >= VERTPREPROC_PARALLEL_MIN_TRIANGLES)
    {
        continue; // in pass 2
    }

    for (uint tri = 0; tri < triCount; tri++)
    {
        VERTPREPROC_TRIANGLE( inst, tri )
    }
}

// pass 2: the very large geometries, their triangles split between all threads
for (uint localGeomIndex = 0; localGeomIndex < geomCount; localGeomIndex++)
{
    const ShGeometryInstance inst = geometryInstances[geomIndexOffset + localGeomIndex];

#if defined(VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE)
    if ((inst.flags & GEOM_INST_FLAG_IS_MOVABLE) == 0)
    {
        continue;
    }
#endif
    if ((inst.flags & GEOM_INST_FLAG_GENERATE_NORMALS) == 0)
    {
        continue;
    }

    const bool  useIndices = inst.baseIndexIndex != UINT32_MAX;
    const float normalSign = float((inst.flags & GEOM_INST_FLAG_INVERTED_NORMALS) == 0) * 2.0 - 1.0;
    const uint  triCount   = useIndices ? inst.indexCount / 3 : inst.vertexCount / 3;

    if (triCount < VERTPREPROC_PARALLEL_MIN_TRIANGLES)
    {
        continue;
    }

    for (uint tri = gl_LocalInvocationID.x; tri < triCount; tri += gl_WorkGroupSize.x)
    {
        VERTPREPROC_TRIANGLE( inst, tri )
    }
}

#undef VERTPREPROC_TRIANGLE
#undef VERTPREPROC_PARALLEL_MIN_TRIANGLES


#undef GET_POSITIONS
#undef GET_NORMALS
#undef SET_NORMALS
#undef INDICES

#undef VERTEX_PREPROCESS_PARTIAL_STATIC_ALL
#undef VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE
#undef VERTEX_PREPROCESS_PARTIAL_DYNAMIC