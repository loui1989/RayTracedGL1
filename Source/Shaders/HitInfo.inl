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

#ifdef DESC_SET_VERTEX_DATA
#ifdef DESC_SET_GLOBAL_UNIFORM
#ifdef DESC_SET_TEXTURES



// Doom64-RT: N64-style three-point albedo filtering (texFilterParams.x == 1).
// This file is included once per HITINFO_INL_* variant, hence the guard.
#ifndef HITINFO_THREEPOINT_DEFINED
#define HITINFO_THREEPOINT_DEFINED

bool albedoThreePoint()
{
    return globalUniform.texFilterParams.x > 0.5;
}

// 'lod' already includes the sampler bias (texFilterParams.y). The quad of four
// texels around 'uv' is split along its diagonal and the three texels of the
// half 'uv' falls in are blended barycentrically -- what the N64's RDP did.
// Sampled at texel centres of one whole mip level; the explicit LOD cancels the
// sampler's own bias so the hardware lands on exactly that level.
vec4 getTextureSampleThreePoint( uint textureIndex, const vec2 uv, float lod )
{
    const int   levels = textureQueryLevels( globalTextures_Get( textureIndex ) );
    const int   level  = clamp( int( floor( lod + 0.5 ) ), 0, max( levels - 1, 0 ) );
    const float hwLod  = float( level ) - globalUniform.texFilterParams.y;

    const vec2 size = vec2( textureSize( globalTextures_Get( textureIndex ), level ) );
    const vec2 inv  = 1.0 / size;
    const vec2 p    = uv * size - 0.5;
    const vec2 f    = fract( p );
    const vec2 c00  = ( floor( p ) + 0.5 ) * inv;

    if( f.x + f.y <= 1.0 )
    {
        const vec4 a = getTextureSampleLod( textureIndex, c00, hwLod );
        const vec4 b = getTextureSampleLod( textureIndex, c00 + vec2( inv.x, 0 ), hwLod );
        const vec4 c = getTextureSampleLod( textureIndex, c00 + vec2( 0, inv.y ), hwLod );
        return a + f.x * ( b - a ) + f.y * ( c - a );
    }
    else
    {
        const vec4 a = getTextureSampleLod( textureIndex, c00 + inv, hwLod );
        const vec4 b = getTextureSampleLod( textureIndex, c00 + vec2( 0, inv.y ), hwLod );
        const vec4 c = getTextureSampleLod( textureIndex, c00 + vec2( inv.x, 0 ), hwLod );
        return a + ( 1.0 - f.x ) * ( b - a ) + ( 1.0 - f.y ) * ( c - a );
    }
}

// Mip level of a textureGrad() footprint, plus the sampler bias.
float threePointLodGrad( uint textureIndex, const vec2 dx, const vec2 dy )
{
    const vec2  size = vec2( getTextureSize( textureIndex ) );
    const vec2  a    = dx * size;
    const vec2  b    = dy * size;
    const float r    = max( dot( a, a ), dot( b, b ) );
    return 0.5 * log2( max( r, 1e-12 ) ) + globalUniform.texFilterParams.y;
}

#endif // HITINFO_THREEPOINT_DEFINED



#if defined( HITINFO_INL_PRIM )
vec3 processAlbedoGrad(         
    uint geometryInstanceFlags, 
    const vec2 texCoords[ TEXLAYER_MAX ], const uint layerColorTextures[ TEXLAYER_MAX ], const uint layerColors[ TEXLAYER_MAX ], 
    const vec2 dPdx[ TEXLAYER_MAX ], const vec2 dPdy[ TEXLAYER_MAX ] )

#elif defined( HITINFO_INL_RFL )
vec3 processAlbedoRayConeDeriv( 
    uint geometryInstanceFlags, 
    const vec2 texCoords[ TEXLAYER_MAX ], const uint layerColorTextures[ TEXLAYER_MAX ], const uint layerColors[ TEXLAYER_MAX ], 
    const DerivativeSet derivSet )

#elif defined( HITINFO_INL_INDIR )
vec3 processAlbedo(             
    uint geometryInstanceFlags, 
    const vec2 texCoords[ TEXLAYER_MAX ], const uint layerColorTextures[ TEXLAYER_MAX ], const uint layerColors[ TEXLAYER_MAX ], 
    float lod )
#endif
{
    #if GEOM_INST_FLAG_BLENDING_LAYER_COUNT != 4
        #error
    #endif
    #if MATERIAL_MAX_ALBEDO_LAYERS > GEOM_INST_FLAG_BLENDING_LAYER_COUNT
        #error
    #endif

    vec3 dst                 = vec3( 1.0 );
    bool hasAnyAlbedoTexture = false;

    [[unroll]] for( int i = 0; i < MATERIAL_MAX_ALBEDO_LAYERS; i++ )
    {
        if( i == 1 )
        {
            if( ( geometryInstanceFlags & GEOM_INST_FLAG_EXISTS_LAYER1 ) == 0 )
            {
                continue;
            }
        }
        else if( i == 2 )
        {
            if( ( geometryInstanceFlags & GEOM_INST_FLAG_EXISTS_LAYER2 ) == 0 )
            {
                continue;
            }
        }
        else if( i == 3 )
        {
            if( ( geometryInstanceFlags & GEOM_INST_FLAG_EXISTS_LAYER3 ) == 0 )
            {
                continue;
            }
        }

    #if defined( HITINFO_INL_CLASSIC_SHADING )
        if( i == MATERIAL_LIGHTMAP_LAYER_INDEX )
        {
            // ignore lightmap layer, if using RT illumination
            continue;
        }
    #endif

        if( layerColorTextures[ i ] != MATERIAL_NO_TEXTURE )
        {
            vec4 texel;
            if( albedoThreePoint() )
            {
    #if defined( HITINFO_INL_PRIM )
                const float tpLod = threePointLodGrad( layerColorTextures[ i ], dPdx[ i ], dPdy[ i ] );
    #elif defined( HITINFO_INL_RFL )
                const float tpLod = threePointLodGrad(
                    layerColorTextures[ i ], vec2( derivSet.u[ i ], 0 ), vec2( 0, derivSet.u[ i ] ) );
    #elif defined( HITINFO_INL_INDIR )
                const float tpLod = lod + globalUniform.texFilterParams.y;
    #endif
                texel = getTextureSampleThreePoint( layerColorTextures[ i ], texCoords[ i ], tpLod );
            }
            else
            {
                texel =
    #if defined( HITINFO_INL_PRIM )
                getTextureSampleGrad( layerColorTextures[ i ], texCoords[ i ], dPdx[ i ], dPdy[ i ] );
    #elif defined( HITINFO_INL_RFL )
                getTextureSampleDerivSet( layerColorTextures[ i ], texCoords[ i ], derivSet, i );
    #elif defined( HITINFO_INL_INDIR )
                getTextureSampleLod( layerColorTextures[ i ], texCoords[ i ], lod );
    #endif
            }
            const vec4 src = unpackUintColor( layerColors[ i ] ) * texel;

            uint layerBlendType = 
                ( geometryInstanceFlags >> ( MATERIAL_BLENDING_TYPE_BIT_COUNT * i ) ) 
                    & MATERIAL_BLENDING_TYPE_BIT_MASK;

            bool opq = ( layerBlendType == MATERIAL_BLENDING_TYPE_OPAQUE );
            bool alp = ( layerBlendType == MATERIAL_BLENDING_TYPE_ALPHA );
            bool add = ( layerBlendType == MATERIAL_BLENDING_TYPE_ADD );
            bool shd = ( layerBlendType == MATERIAL_BLENDING_TYPE_SHADE );

            // simple fix for layerColorTextures that have alpha-tested blending for the first layer
            // (just makes "opq" instead of "alp" for that partuicular case);
            // without this fix, alpha-tested geometry will have white color around borders
            opq = opq || ( alp && i == 0 );
            alp = alp && !opq;

            dst = float(opq) * (src.rgb) +
                  float(alp) * (src.rgb * src.a + dst * (1 - src.a)) + 
                  float(add) * (src.rgb + dst) +
                  float(shd) * (src.rgb * dst * 2);

            hasAnyAlbedoTexture = true;
        }
    }

    // if no albedo textures, use primary color
    dst = mix( unpackUintColor( layerColors[ 0 ] ).rgb, dst, float( hasAnyAlbedoTexture ) );

    return clamp( dst, vec3( 0 ), vec3( 1 ) );
}


#if defined( HITINFO_INL_PRIM )
// "Ray Traced Reflections in 'Wolfenstein: Youngblood'", Jiho Choi, Jim Kjellin, Patrik Willbo, Dmitry Zhdan
float getBounceLOD(float roughness, float viewDist, float hitDist, float screenWidth, float bounceMipBias)
{    
    const float range = 300.0 * pow((1.0 - roughness) * 0.9 + 0.1, 4.0);

    vec2 f = vec2(viewDist, hitDist);
    f = clamp(f / range, vec2(0.0), vec2(1.0));
    f = sqrt(f);

    float mip = max(log2(3840.0 / screenWidth), 0.0);

    mip += f.x * 10.0;
    mip += f.y * 10.0;

    return mip + bounceMipBias;
}
#endif // HITINFO_INL_PRIM


#if defined(HITINFO_INL_PRIM)
// Fast, Minimum Storage Ray-Triangle Intersection, Moller, Trumbore
vec3 intersectRayTriangle(const mat3 positions, const vec3 orig, const vec3 dir)
{
    const vec3 edge1 = positions[1] - positions[0];
    const vec3 edge2 = positions[2] - positions[0];

    const vec3 pvec = cross(dir, edge2);

    const float det = dot(edge1, pvec);
    const float invDet = 1.0 / det;

    const vec3 tvec = orig - positions[0];
    const vec3 qvec = cross(tvec, edge1);

    const float u = dot(tvec, pvec) * invDet;
    const float v = dot(dir, qvec) * invDet;

    return vec3(1 - u - v, u, v);
}

// 0.0 - deepest point; 1.0 - surface level
float sampleHeightMap( const uint textureIndex, const vec2 texCoords )
{
    return getTextureSampleLod( textureIndex, texCoords, 0 ).r;
}

// Doom64-RT: noise for the liquid flow. PROCEDURAL, and that is the fix, not a
// style choice: the first advection sampled the water normal map's X channel,
// and a normal map's X hugs 0.5 by construction -- the pattern moved and the
// modulation it carried was a few percent, which is invisible at any speed.
// (getLavaHeat samples the same channels; "the lava flow does not do much" is
// almost certainly the same disease.) Value noise on a 256-periodic lattice:
// periodic so the scrolled coordinate can wrap with NO seam, which also keeps
// the hash's fract() inputs small enough for float32 over a long session.
float d64_flowHash( vec2 cell )
{
    cell = fract( cell * vec2( 0.1031, 0.1972 ) );
    cell += dot( cell, cell.yx + 33.33 );
    return fract( ( cell.x + cell.y ) * cell.x );
}

float d64_flowNoise( vec2 p )
{
    vec2 i = mod( floor( p ), 256.0 );
    vec2 f = fract( p );
    vec2 s = f * f * ( 3.0 - 2.0 * f );

    float a = d64_flowHash( i );
    float b = d64_flowHash( mod( i + vec2( 1, 0 ), 256.0 ) );
    float c = d64_flowHash( mod( i + vec2( 0, 1 ), 256.0 ) );
    float d = d64_flowHash( mod( i + vec2( 1, 1 ), 256.0 ) );

    return mix( mix( a, b, s.x ), mix( c, d, s.x ), s.y );
}

const int ParallaxLinearSteps       = 10;
const int ParallaxBinarySearchSteps = 4;

vec2 parallaxTexCoords( const uint  heightTextureIndex,
                        const vec2  texCoords,
                        const vec3  viewDir,
                        const float maxDepth )
{
    const float deltaLayerDepth     = maxDepth / ParallaxLinearSteps;
    const vec2  deltaLayerTexCoords = viewDir.xy / viewDir.z * deltaLayerDepth;

    float depth         = 0.0;
    float curLayerDepth = 0.0;

    // linear search for a layer
    for( int i = 0; i < ParallaxLinearSteps; i++ )
    {
        curLayerDepth         = i * deltaLayerDepth;
        vec2 curLayerTexCoord = texCoords - i * deltaLayerTexCoords;

        depth = maxDepth * ( 1.0 - sampleHeightMap( heightTextureIndex, curLayerTexCoord ) );
        if( depth < curLayerDepth )
        {
            break;
        }
    }

    // binary search for more precise hit position in the layer
    float lower  = curLayerDepth - deltaLayerDepth;
    float higher = curLayerDepth;
    for( int i = 0; i < ParallaxBinarySearchSteps; i++ )
    {
        float mid = ( lower + higher ) * 0.5;
        if( depth < mid )
        {
            higher = mid;
        }
        else
        {
            lower = mid;
        }
    }

    float hit = ( lower + higher ) * 0.5;
    return texCoords - viewDir.xy / viewDir.z * hit;
}

// Correct 'candidateNormal' that was produced by a normal map, or vertex interpolation:
// prevent self intersections, i.e. the reflection over 'candidateNormal' would not
// point into the surface (defined by 'triangleNormal'), which would
// produce zero later in a pipeline, e.g. max(0,dot(n,l))
// "The Iray Light Transport Simulation and Rendering System",
// "A.3 Local Shading Normal Adaption"
vec3 sanitizeNormal( const vec3 triangleNormal, //
                     const vec3 candidateNormal,
                     const vec3 fromViewer )
{
    const vec3 candidateRefl = reflect( fromViewer, candidateNormal );

    float nr = dot( candidateRefl, triangleNormal );

    // if reflection doesn't cause self intersection
    if( nr > 0 )
    {
        return candidateNormal;
    }

    const float a = -nr * 2;
    const float b = max( 0.001, dot( candidateNormal, triangleNormal ) );

    const vec3 sanitizedRefl = candidateRefl + a / b * candidateNormal;

    // r = v - 2 * (v.n) * n
    // n ~ -r + v
    return -safeNormalize2( -normalize( sanitizedRefl ) + fromViewer, //
                            triangleNormal );
}
#endif // HITINFO_INL_PRIM


#if defined(HITINFO_INL_PRIM)

ShHitInfo getHitInfoPrimaryRay(
    const ShPayload pl,
    const vec3 rayOrigin, const vec3 viewDir, const vec3 rayDirAX, const vec3 rayDirAY,
    out vec2 motion, out float motionDepthLinear,
    out vec2 gradDepth, out float depthNDC, out float depthLinear,
    out vec3 emission,
    // Doom64-RT: the liquid FLOW DETAIL, 0..1 -- a noise texture advected along
    // the vein direction baked into the height map's .g/.b. It has to be built
    // HERE and handed onward: the stylized liquid surface is shaded in the
    // refl/refr pass, which restores its hit from the G-buffer and has no
    // ShTriangle, so it can never sample a material texture of its own.
    // Exactly 0 means "no flow here" (not a liquid, no height map, or a texel
    // the bake left still).
    out float liquidFlow)

#elif defined(HITINFO_INL_RFL)

ShHitInfo getHitInfoWithRayCone_ReflectionRefraction(
    const ShPayload pl, const RayCone rayCone,
    const vec3 rayOrigin, const vec3 rayDir, const vec3 viewDir,
    in out vec3 virtualPosForMotion,
    out float rayLen,
    out vec2 motion, out float motionDepthLinear,
    out vec3 emission,
    out bool hasNormalMap)

#elif defined(HITINFO_INL_INDIR)

ShHitInfo getHitInfoBounce(
    const ShPayload pl, const vec3 rayOrigin, float originRoughness, float bounceMipBias,
    out vec3 emission)

#endif
{
    ShHitInfo h;

#if defined( HITINFO_INL_PRIM )
    liquidFlow = 0.0;
#endif

    int instanceId, instCustomIndex;
    int geomIndex, primIndex;

    unpackInstanceIdAndCustomIndex(pl.instIdAndIndex, instanceId, instCustomIndex);
    unpackGeometryAndPrimitiveIndex(pl.geomAndPrimIndex, geomIndex, primIndex);

    const ShTriangle tr = getTriangle(instanceId, instCustomIndex, geomIndex, primIndex);



    // GEOMETRY

    const vec2 inBaryCoords = pl.baryCoords;
    const vec3 baryCoords = vec3(1.0f - inBaryCoords.x - inBaryCoords.y, inBaryCoords.x, inBaryCoords.y);

     vec2 texCoords[] = {
        tr.layerTexCoord[ 0 ] * baryCoords,
#if !SUPPRESS_TEXLAYERS
        tr.layerTexCoord[ 1 ] * baryCoords,
        tr.layerTexCoord[ 2 ] * baryCoords,
        tr.layerTexCoord[ 3 ] * baryCoords,
#endif
    };

    h.hitPosition = tr.positions * baryCoords;

    vec3 exactNormal = safeNormalize3(        //
        cross( tr.positions[ 1 ] - tr.positions[ 0 ], //
               tr.positions[ 2 ] - tr.positions[ 0 ] ),
        0.00000001,
        vec3( 0, 1, 0 ) );
    {
        if( ( tr.geometryInstanceFlags & GEOM_INST_FLAG_EXACT_NORMALS ) == 0 )
        {
            h.normal = normalize( tr.normals * baryCoords );
        }
        else
        {
            h.normal = exactNormal;
        }

        // always face ray origin
        const vec3 toViewer = safeNormalize3( rayOrigin - h.hitPosition, 0.00000001, vec3( 0 ) );
        if( dot( exactNormal, toViewer ) < 0 )
        {
            exactNormal *= -1;
            h.normal *= -1;
        }

        h.normal = sanitizeNormal( exactNormal, h.normal, -toViewer );
    }



    // CLIP SPACE

#if defined(HITINFO_INL_PRIM)
    // Tracing Ray Differentials, Igehy
    // instead of casting new rays, check intersections on the same triangle
    const vec3 baryCoordsAX = intersectRayTriangle(tr.positions, rayOrigin, rayDirAX);
    const vec3 baryCoordsAY = intersectRayTriangle(tr.positions, rayOrigin, rayDirAY);

    const vec4 viewSpacePosCur   = globalUniform.view     * vec4(h.hitPosition, 1.0);
    const vec4 viewSpacePosPrev  = globalUniform.viewPrev * vec4(tr.prevPositions * baryCoords, 1.0);
    const vec4 viewSpacePosAX    = globalUniform.view     * vec4(tr.positions     * baryCoordsAX, 1.0);
    const vec4 viewSpacePosAY    = globalUniform.view     * vec4(tr.positions     * baryCoordsAY, 1.0);

    const vec4 clipSpacePosCur   = globalUniform.projection     * viewSpacePosCur;
    const vec4 clipSpacePosPrev  = globalUniform.projectionPrev * viewSpacePosPrev;

    const float clipSpaceDepth   = clipSpacePosCur[2];
    const float clipSpaceDepthAX = dot(globalUniform.projection[2], viewSpacePosAX);
    const float clipSpaceDepthAY = dot(globalUniform.projection[2], viewSpacePosAY);

    const vec3 ndcCur            = clipSpacePosCur.xyz  / clipSpacePosCur.w;
    const vec3 ndcPrev           = clipSpacePosPrev.xyz / clipSpacePosPrev.w;

    const vec2 screenSpaceCur    = ndcCur.xy  * 0.5 + 0.5;
    const vec2 screenSpacePrev   = ndcPrev.xy * 0.5 + 0.5;
#endif // HITINFO_INL_PRIM

#if defined(HITINFO_INL_RFL) 
    rayLen = length(h.hitPosition - rayOrigin);
    virtualPosForMotion += viewDir * rayLen;

    const vec4 viewSpacePosCur   = globalUniform.view     * vec4(virtualPosForMotion, 1.0);
    const vec4 viewSpacePosPrev  = globalUniform.viewPrev * vec4(virtualPosForMotion, 1.0);
    const vec4 clipSpacePosCur   = globalUniform.projection     * viewSpacePosCur;
    const vec4 clipSpacePosPrev  = globalUniform.projectionPrev * viewSpacePosPrev;
    const vec3 ndcCur            = clipSpacePosCur.xyz  / clipSpacePosCur.w;
    const vec3 ndcPrev           = clipSpacePosPrev.xyz / clipSpacePosPrev.w;
    const vec2 screenSpaceCur    = ndcCur.xy  * 0.5 + 0.5;
    const vec2 screenSpacePrev   = ndcPrev.xy * 0.5 + 0.5;

    const float clipSpaceDepth   = clipSpacePosCur[2];
#endif // HITINFO_INL_RFL

#if defined(HITINFO_INL_PRIM)
    depthNDC = ndcCur.z;
    depthLinear = length(viewSpacePosCur.xyz);
#endif



    // MOTION

#if defined(HITINFO_INL_PRIM) || defined(HITINFO_INL_RFL)
    // difference in screen-space
    motion = (screenSpacePrev - screenSpaceCur);
#endif

#if defined(HITINFO_INL_PRIM)
    motionDepthLinear = length(viewSpacePosPrev.xyz) - depthLinear;
#elif defined(HITINFO_INL_RFL)
    motionDepthLinear = length(viewSpacePosPrev.xyz) - length(viewSpacePosCur.xyz);
#endif 

#if defined(HITINFO_INL_PRIM) 
    // gradient of clip-space depth with respect to clip-space coordinates
    gradDepth = vec2(clipSpaceDepthAX - clipSpaceDepth, clipSpaceDepthAY - clipSpaceDepth);
#elif defined(HITINFO_INL_RFL)
    // don't touch gradDepth for reflections / refractions
#endif



    // TEXTURE SAMPLING

#if defined( HITINFO_INL_PRIM )
    // pixel's footprint in texture space
    const vec2 dTdx[] = {
        ( tr.layerTexCoord[ 0 ] * baryCoordsAX - texCoords[ 0 ] ),
#if !SUPPRESS_TEXLAYERS
        ( tr.layerTexCoord[ 1 ] * baryCoordsAX - texCoords[ 1 ] ),
        ( tr.layerTexCoord[ 2 ] * baryCoordsAX - texCoords[ 2 ] ),
        ( tr.layerTexCoord[ 3 ] * baryCoordsAX - texCoords[ 3 ] ),
#endif
    };

    const vec2 dTdy[] = {
        ( tr.layerTexCoord[ 0 ] * baryCoordsAY - texCoords[ 0 ] ),
#if !SUPPRESS_TEXLAYERS
        ( tr.layerTexCoord[ 1 ] * baryCoordsAY - texCoords[ 1 ] ),
        ( tr.layerTexCoord[ 2 ] * baryCoordsAY - texCoords[ 2 ] ),
        ( tr.layerTexCoord[ 3 ] * baryCoordsAY - texCoords[ 3 ] ),
#endif
    };
#elif defined(HITINFO_INL_RFL)
    const DerivativeSet derivSet = getTriangleUVDerivativesFromRayCone(tr, h.normal, rayCone, rayDir);
#endif



    // HEIGHT / NORMAL MAP (ignored for indirect)
    // materialStripFlags: bit0=N, bit1=emis, bit2=metallic, bit3=H, bit4=roughness

    // Doom64-RT: a BILLBOARD is one quad with one normal, so its indirect
    // specular reflection vector is identical for every texel and the whole
    // sprite takes one flat colour from the room. That failure has no wall
    // equivalent, so sprites carry their own dials -- and spritePbr 0 is the
    // master off switch for the entire sprite material pass, dropping the
    // authored _orm and the derived _n in one go.
    const bool isSprite  = ( tr.geometryInstanceFlags & GEOM_INST_FLAG_SPRITE ) != 0;
    // spritePbr is a MIX, not a switch: 1 = the material exactly as authored,
    // 0 = the plain dielectric a sprite had before any labelling. Everything
    // between dials the whole sprite pass toward neutral, which does two things
    // at once -- it lowers how hard sprites react to light, and it COMPRESSES
    // the spread between classes, because every class converges on the same
    // neutral as the mix falls. A checkerboard of 0.2 metal beside 0.9 rough
    // flesh stops being a contrast problem when both are 80% of the way to the
    // same place.
    const float spriteMix = clamp( globalUniform.spritePbr, 0.0, 1.0 );
    const bool  spriteOff = isSprite && spriteMix < 0.001;

    const bool stripNormals    = ( globalUniform.materialStripFlags & 1u ) != 0 || spriteOff;
    const bool stripEmissives  = ( globalUniform.materialStripFlags & 2u ) != 0;
    const bool stripMetallic   = ( globalUniform.materialStripFlags & 4u ) != 0;
    const bool stripHeight     = ( globalUniform.materialStripFlags & 8u ) != 0 || spriteOff;
    const bool stripRoughness  = ( globalUniform.materialStripFlags & 16u ) != 0;

#if defined( HITINFO_INL_PRIM ) || defined( HITINFO_INL_RFL )
    vec3 tangent, bitangent;
    const bool needTbn =
        ( !stripHeight && tr.heightTexture != MATERIAL_NO_TEXTURE ) ||
        ( !stripNormals && tr.normalTexture != MATERIAL_NO_TEXTURE );
    if( needTbn )
    {
        makeTangentBitangent( tr.positions, tr.layerTexCoord[ 0 ], tangent, bitangent );
    }

    if( !stripHeight && globalUniform.parallaxMaxDepth > 0.0001 &&
        tr.heightTexture != MATERIAL_NO_TEXTURE )
    {
        vec3 viewDirInTextureSpace;
        {
            mat3 worldToTbn       = transpose( mat3( tangent, bitangent, h.normal ) );
            viewDirInTextureSpace = normalize( worldToTbn * viewDir );
        }

        texCoords[ 0 ] = parallaxTexCoords( tr.heightTexture,
                                            texCoords[ 0 ],
                                            viewDirInTextureSpace,
                                            globalUniform.parallaxMaxDepth );
    }

#if defined( HITINFO_INL_PRIM )
    // Doom64-RT: the flow map. The height map's .g/.b carry the vein's tangent
    // DIRECTION in texture space (sampleHeightMap() reads .r and nothing else,
    // so they are free on a map we already author, and they are registered
    // with the albedo, parallax shift included -- this runs AFTER the block
    // above, so the flow follows the same displaced texel the colour does).
    //
    // The detail is sampled in a VEIN-ALIGNED frame: u runs along the channel
    // and SCROLLS, v runs across it at liquidFlowAspect times the frequency.
    // That turns the noise into elongated streaks sliding lengthwise down the
    // vein, which is the strongest "liquid running" cue there is. Two earlier
    // versions failed short of it and are worth remembering:
    //   1. a brightness band on a baked phase -- nothing in the picture is
    //      displaced, the eye reads flicker;
    //   2. round blobs advected from base UV with a ping-pong cross-fade --
    //      isotropic blobs at vein scale read as shimmer, and the two blended
    //      copies soften what little motion there was.
    // No ping-pong here, and none needed: it existed to bound the distortion
    // of an offset-from-base advection, but scrolling the wrapping noise's OWN
    // coordinate is well-defined forever and never resets. Where the direction
    // varies along a run the frame shears the noise slightly -- fine, liquid
    // shears.
    //
    // The direction is a VECTOR, not an angle: bilinear filtering between two
    // disagreeing texels shrinks it toward zero, so the flow FADES at a
    // junction or a sign flip instead of tearing. Length gates it.
    if( ( tr.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_WATER ) != 0 &&
        tr.heightTexture != MATERIAL_NO_TEXTURE )
    {
        vec2 dir =
            getTextureSampleLod( tr.heightTexture, texCoords[ 0 ], 0 ).gb * 2.0 - vec2( 1.0 );
        const float len = length( dir );
        // 0.15: above the noise of two blended texels. Cells bake to (0,0).
        if( len > 0.15 )
        {
            dir /= len;
            const vec2 perp = vec2( -dir.y, dir.x );

            // liquidFlowScale: detail tiles per liquid tile along the vein.
            // liquidFlowSpeed: detail tiles scrolled per second.
            const float u = dot( texCoords[ 0 ], dir ) * globalUniform.liquidFlowScale -
                            globalUniform.time * globalUniform.liquidFlowSpeed;
            const float v = dot( texCoords[ 0 ], perp ) * globalUniform.liquidFlowScale *
                            globalUniform.liquidFlowAspect;

            // Procedural, full-range noise -- see d64_flowNoise for why the
            // water normal map could not be the source. mod() is seamless
            // because the lattice itself is 256-periodic.
            float d = d64_flowNoise( vec2( mod( u, 256.0 ), v ) );

            // shape into clots: plateaus of bright and dark with fast
            // transitions, so what slides down the vein reads as blobs of
            // liquid rather than as smooth static
            d = smoothstep( 0.30, 0.70, d );

            // nudged off zero so that EXACTLY 0 keeps meaning "no flow here"
            liquidFlow = d * 0.998 + 0.001;
        }
    }
#endif


    if( !stripNormals && tr.normalTexture != MATERIAL_NO_TEXTURE )
    {
        vec2 nrm =
    #if defined( HITINFO_INL_PRIM )
            getTextureSampleGrad( tr.normalTexture, texCoords[ 0 ], dTdx[ 0 ], dTdy[ 0 ] )
    #elif defined( HITINFO_INL_RFL )
            getTextureSampleDerivSet( tr.normalTexture, texCoords[ 0 ], derivSet, 0 )
    #endif
            .xy;
        nrm.xy = nrm.xy * 2.0 - vec2( 1.0 );

        const vec3 newNormal = safeNormalize2( tangent * nrm.x + bitangent * nrm.y + h.normal, //
                                               h.normal );
        const float nmStren =
            globalUniform.normalMapStrength *
            ( isSprite ? globalUniform.spriteNormalStrength * spriteMix : 1.0 );
        h.normal = safeNormalize2( mix( h.normal, newNormal, nmStren ), //
                                   h.normal );

        const vec3 toViewer = safeNormalize2( rayOrigin - h.hitPosition, vec3( 0 ) );
        h.normal            = sanitizeNormal( exactNormal, h.normal, -toViewer );
    }

#if defined( HITINFO_INL_RFL )
    hasNormalMap = ( !stripNormals && tr.normalTexture != MATERIAL_NO_TEXTURE );
#endif

#endif // HITINFO_INL_PRIM || HITINFO_INL_RFL



    // ALBEDO

#if defined( HITINFO_INL_PRIM )
    h.albedo = processAlbedoGrad( tr.geometryInstanceFlags, 
                                  texCoords,
                                  tr.layerColorTextures, 
                                  tr.layerColors, 
                                  dTdx, dTdy );
#elif defined(HITINFO_INL_RFL)
    h.albedo = processAlbedoRayConeDeriv( tr.geometryInstanceFlags,
                                          texCoords,
                                          tr.layerColorTextures,
                                          tr.layerColors,
                                          derivSet );
#elif defined(HITINFO_INL_INDIR)
    const float viewDist = length(h.hitPosition - globalUniform.cameraPosition.xyz);
    const float hitDistance = length(h.hitPosition - rayOrigin);

    const float lod = getBounceLOD(originRoughness, viewDist, hitDistance, globalUniform.renderWidth, bounceMipBias);

    h.albedo = processAlbedo(tr.geometryInstanceFlags, texCoords, tr.layerColorTextures, tr.layerColors, lod);
#endif



    // METALLIC-ROUGHNESS

    if( tr.occlusionRougnessMetallicTexture != MATERIAL_NO_TEXTURE )
    {
        const vec3 orm =
    #if defined( HITINFO_INL_PRIM )
            getTextureSampleGrad( tr.occlusionRougnessMetallicTexture, texCoords[ 0 ], dTdx[ 0 ], dTdy[ 0 ] ).xyz;
    #elif defined( HITINFO_INL_RFL )
            getTextureSampleDerivSet( tr.occlusionRougnessMetallicTexture, texCoords[ 0 ], derivSet, 0 ).xyz;
    #elif defined( HITINFO_INL_INDIR )
            getTextureSampleLod( tr.occlusionRougnessMetallicTexture, texCoords[ 0 ], lod ).xyz;
    #endif

        h.roughness = stripRoughness ? 1.0 : orm[ 1 ];
        h.metallic  = stripMetallic ? 0.0 : orm[ 2 ];
        // a polished texture's _orm roughness is its mirror mask: smooth texels mirror, rough
        // ones (0.7+) don't; roughness 0 mirrors ~3x as much as the old uniform polish
        h.polish    = clamp( ( 0.7 - orm[ 1 ] ) * 4.5, 0.0, 3.0 );
    }
    else
    {
        h.roughness = stripRoughness ? 1.0 : tr.roughnessDefault;
        h.metallic  = stripMetallic ? 0.0 : tr.metallicDefault;
        h.polish    = 1.0;
    }
    // Dev Materials: soft path between authored roughness and full matte (Strip roughness = hard 1).
    h.roughness = mix( h.roughness, 1.0, clamp( globalUniform.materialRoughnessTowardMatte, 0.0, 1.0 ) );
    h.roughness = max( h.roughness, max( globalUniform.minRoughness, MIN_GGX_ROUGHNESS ) );

    // Doom64-RT: fail-safes over hand-labelled metalness.
    //
    // A metal has no diffuse lobe -- it only shows a blurred image of whatever is
    // around it. In a dark interior that is nothing, so a surface wrongly called
    // metal goes black, and the rougher it is the worse it gets: the same small
    // energy is spread over more of the hemisphere. Raising minRoughness cannot
    // help, because that is the direction that makes it worse.
    //
    //   metallicRoughCut  roughness at which metalness starts fading out, over a
    //                     metallicRoughBand-wide band. Very rough "metal" is
    //                     usually oxide, paint or scale -- all dielectrics -- so
    //                     this is physical as well as a safety net. 0 disables.
    //   metallicMax       hard ceiling on metalness, so every metal keeps at
    //                     least (1 - metallicMax) of its diffuse response and can
    //                     never reach pure black. 1 disables.
    if( globalUniform.metallicRoughCut > 0.0 )
    {
        const float band = max( globalUniform.metallicRoughBand, 0.001 );
        h.metallic *= 1.0 - smoothstep( globalUniform.metallicRoughCut,
                                        globalUniform.metallicRoughCut + band,
                                        h.roughness );
    }
    h.metallic = min( h.metallic, globalUniform.metallicMax );

    // WALLS AND FLATS take the same treatment from their own dial. Kept
    // separate from the sprite one because the two fail differently: a wall has
    // per-texel normals and never shows the flat-quad wash, so its reason to be
    // dialled back is noise and taste rather than a rendering artefact.
    if( !isSprite )
    {
        const float worldMix = clamp( globalUniform.worldPbr, 0.0, 1.0 );
        h.metallic  = mix( 0.0, h.metallic,  worldMix );
        h.roughness = mix( 1.0, h.roughness, worldMix );
    }

    // The sprite set, on top of the world clamps rather than instead of them.
    if( isSprite )
    {
        if( spriteOff )
        {
            h.metallic  = 0.0;
            h.roughness = 1.0;
        }
        else
        {
            float m = min( h.metallic, globalUniform.spriteMetallicMax );
            // A FLOOR, and the direct lever on the beige wash: a broader lobe
            // spreads that single room reflection instead of mirroring it.
            float r = max( h.roughness, globalUniform.spriteRoughMin );

            // NEUTRAL IS metallic 0 / roughness 1 because that is exactly what
            // a sprite rendered as before any of this existed -- an entry with
            // no _orm and no roughnessDefault. So spriteMix 0 is not an
            // approximation of the old look, it IS the old look.
            h.metallic  = mix( 0.0, m, spriteMix );
            h.roughness = mix( 1.0, r, spriteMix );
        }
    }



    // EMISSIVE
    //
    // With an _e map:
    //   - Primary / reflection: raw _e (on-screen glow via emissionMaxScreenColor).
    //   - Indirect: _e * emissiveMult, then emissionMapBoost in RtRaygenIndirect.
    // Without _e: albedo * emissiveMult (all paths), as before.

    if( stripEmissives )
    {
        emission = vec3( 0.0 );
    }
    else if( tr.emissiveTexture != MATERIAL_NO_TEXTURE )
    {
        emission =
    #if defined( HITINFO_INL_PRIM )
            getTextureSampleGrad( tr.emissiveTexture, texCoords[ 0 ], dTdx[ 0 ], dTdy[ 0 ] ).rgb;
    #elif defined( HITINFO_INL_RFL )
            getTextureSampleDerivSet( tr.emissiveTexture, texCoords[ 0 ], derivSet, 0 ).rgb;
    #elif defined( HITINFO_INL_INDIR )
            getTextureSampleLod( tr.emissiveTexture, texCoords[ 0 ], lod ).rgb;
    #endif
    #if defined( HITINFO_INL_INDIR )
        // Doom64-RT: a SCREEN_SCALED primitive's emissiveMult is what it LOOKS
        // like; what it feeds the GI is a SEPARATE per-primitive value, because
        // the two are animated on different clocks. The painted bulbs are held
        // back to meet the light they cast (it arrives late through the
        // denoiser history); the light itself must not be held back with them,
        // or every frame of delay added to the bulbs moves the pool by the same
        // amount and the gap never closes. A constant GI was tried in between
        // and it was worse: the "real light turning" WAS this GI, so freezing
        // it froze the chase. See RgMeshPrimitiveInfo::emissiveGi.
        if( ( tr.geometryInstanceFlags & GEOM_INST_FLAG_EMIS_SCREEN_SCALED ) != 0 )
        {
            emission *= tr.emissiveMultGi;
        }
        else
        {
            emission *= tr.emissiveMult;
        }
    #else
        // Doom64-RT: opt-in, so a fixture that is meant to SWITCH can.
        //
        // The rule above -- raw _e on screen, emissiveMult on the indirect path
        // only -- means a caller can change how much a painted lamp feeds the GI
        // and cannot change how bright it looks. For a fixture whose whole job is
        // to turn on and off, that is the one thing it needs. MAP01's pre-exit
        // pillar sweeps a light around four bulb panels; without this the painted
        // bulbs stay lit through the entire cycle.
        //
        // One bit per primitive rather than a global change of the rule: every
        // other emissive in the game was balanced against raw _e and must stay
        // there. See RG_MESH_PRIMITIVE_EMISSIVE_SCREEN_SCALED.
        //
        // On screen, emissiveMult is applied AS IS. The GI above does not use
        // it at all for these primitives; it uses the constant. That split is
        // the whole point: the screen value is animated (a bulb switching) and
        // the GI must not follow it, or the fixture's bounce light chases its
        // own bulbs and the two can never be lined up -- every frame of delay
        // added to the bulbs moved the pool of light back by the same amount.
        if( ( tr.geometryInstanceFlags & GEOM_INST_FLAG_EMIS_SCREEN_SCALED ) != 0 )
        {
            emission *= tr.emissiveMult;
        }
    #endif
    }
    else
    {
        emission = h.albedo * tr.emissiveMult;
    }

    // Doom64-RT: lava as an AREA light source.
    //
    // Applied on the INDIRECT path only, which is the whole point: a lake lights
    // a room by being a large warm surface, not by a grid of point lights, and
    // the grid version leaves a visible circle on the wall as the player walks
    // past each one. Indirect emission is already the mechanism for that
    // (emissiveMult here, emissionMapBoost in RtRaygenIndirect); it just needs
    // to be strong enough on this one material.
#if defined( HITINFO_INL_INDIR )
    if( ( tr.geometryInstanceFlags & GEOM_INST_FLAG_LAVA ) != 0 )
    {
        // Tinted here too, or the room bounces a different colour than the
        // surface it came off -- the giveaway that the hue is a screen effect.
        emission *= globalUniform.lavaGiBoost * globalUniform.lavaTint.rgb;
    }
#endif



    // MISC

    h.instCustomIndex = instCustomIndex;
    h.geometryInstanceFlags = tr.geometryInstanceFlags;
    h.portalIndex = tr.portalIndex;

    return h;
}

#endif // DESC_SET_TEXTURES
#endif // DESC_SET_GLOBAL_UNIFORM
#endif // DESC_SET_VERTEX_DATA