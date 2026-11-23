// Doom64-RT: LOCALISED SMOKE in the froxel volume.
//
// The volumetric medium (Volumetric.h, RtVolumetric.rgen) is ONE density for
// the whole level and the grid is camera-fitted, so there is no way to say
// "there is smoke HERE". This adds a small list of world-space spheres whose
// density is ADDED to that medium, evaluated per froxel.
//
// Everything downstream is untouched. CmVolumetricProcess.comp is a straight
// front-to-back prefix sum over whatever the raygen wrote, so a puff gets
// correct occlusion and transmittance for free -- it darkens what is behind it
// and is darkened by fog in front of it, with no second pass and no extra rays.
//
// THE PROPERTY THIS FILE HAS TO KEEP: with smokeCount 0 every function here
// returns zero, and the caller's arithmetic collapses back to exactly the fog's
// two lines. The fog is shipped and tuned against a measured transmittance
// ladder (docs/rt-fog.md); smoke may not move it. See the collapse note at
// smoke_blendTint below.
//
// The puffs ride in the global uniform rather than a storage buffer: at
// SMOKE_PUFF_MAX 128 that is 6 KB in a UBO that already carries
// viewProjCubemap, against a whole descriptor set and a per-frame staging copy
// for a LightManager-shaped buffer. 128 takes the whole ShGlobalUniform to 8160
// bytes, still under the 16384 the Vulkan spec guarantees for
// maxUniformBufferRange -- which is the ceiling on this approach. If the budget
// ever needs hundreds, THAT is when the storage buffer earns its complexity.
//
// The cap and the BUDGET are different numbers and are limited by different
// things. The cap is uniform bytes. The budget -- how many are uploaded -- is
// shader time, because smoke_evalAt below runs per froxel; that is what the
// early reject in it exists to flatten.

#ifndef SMOKE_H_
#define SMOKE_H_

// Accumulated smoke at a point.
//   .rgb -- albedo premultiplied by density, i.e. sum( d_i * albedo_i )
//   .a   -- density,                        i.e. sum( d_i )
// Both zero when there is no smoke, which is the case every frame on every map
// until someone fires a gun.
vec4 smoke_evalAt( vec3 worldPos )
{
    vec4 acc = vec4( 0.0 );

    for( uint i = 0; i < globalUniform.smokeCount; i++ )
    {
        const vec4 puff = globalUniform.smokePuffs[ i ];

        // ANISOTROPIC, and the reason is the grid, not the art. At 1.5 m a
        // froxel is 1.7 cm across the screen but 47 cm deep, so a SPHERE thin
        // enough to read as a filament falls between depth slices and renders
        // nothing. Splitting the offset into along-view and across-view parts
        // and dividing each by its own radius gives a puff that is centimetres
        // wide where you can see it and half a slice deep where you cannot.
        //
        // puff.w   = radius ALONG the view (grid-safe, set engine-side)
        // shape.x  = radius ACROSS the view (what the profile actually asked for)
        const vec4  shape  = globalUniform.smokeShape[ i ];
        const float rAlong = max( puff.w, 0.0001 );
        const float rPerp  = max( shape.x, 0.0001 );

        // PIXEL-ART STYLIZATION, part one: snap the sample to a world voxel.
        //
        // WORLD space, not screen. Screen-space blocks are the obvious way to
        // get "pixels" and they crawl as soon as the camera turns, which the
        // eye reads as noise rather than as style -- the sprites this is
        // imitating are pixel grids that stay put on the OBJECT. Snapping the
        // evaluation position instead gives the puff hard voxel steps that are
        // stable in the level and move with the smoke.
        vec3 sampleAt = worldPos;
        if( globalUniform.smokeStylizeGrid > 0.0 )
        {
            const float g = globalUniform.smokeStylizeGrid;
            sampleAt      = ( floor( worldPos / g ) + 0.5 ) * g;
        }

        const vec3 d = sampleAt - puff.xyz;

        // CHEAP REJECT FIRST, and this is what makes a big budget affordable.
        //
        // This loop runs for every uploaded puff in each of ~900k froxels, and
        // almost every one of those pairs is a miss -- a puff is centimetres to
        // a couple of metres across in a volume tens of metres deep. The
        // ellipsoid is contained in the sphere of radius max(rAlong,rPerp), so
        // testing that sphere first is exact, not an approximation, and it costs
        // one dot product against the sqrt and the divisions below. The loop's
        // cost becomes "puffs near THIS cell" rather than "puffs uploaded",
        // which is what lets the budget grow for ambient sources at all.
        const float rMax = max( rAlong, rPerp );
        if( dot( d, d ) > rMax * rMax )
        {
            continue;
        }

        // Precomputed engine-side, once per puff per frame: this used to be
        // normalize( puff.xyz - cameraPosition ) evaluated per froxel.
        const vec3  vdir  = shape.yzw;
        const float along = dot( d, vdir );
        const vec3  perp  = d - along * vdir;

        const float t = length( vec2( along / rAlong, length( perp ) / rPerp ) );

        if( t < 1.0 )
        {
            // ( 1 - t^2 )^2: 1 at the core, 0 with a zero derivative at the
            // rim, so a puff crossing a froxel boundary does not step. Cheaper
            // than a gaussian and, unlike one, actually reaches zero at the
            // radius instead of leaving a faint halo the size of the budget.
            float w = ( 1.0 - t * t );
            w *= w;

            // PIXEL-ART STYLIZATION, part two: posterize that falloff.
            //
            // The smooth curve above is physically right and reads as an
            // airbrushed blob against art that is entirely hard-edged pixels.
            // Quantizing it into a few bands gives the stepped, banded edge the
            // sprites have, and it is the half of this effect that does the
            // most work -- a puff reads as drawn rather than as rendered.
            //
            // CEIL, not round or floor. The outermost band is where the puff
            // meets the world, and flooring it quantizes the rim to zero: the
            // puff would shrink by a full band and its silhouette would come
            // back soft, which is the thing being fixed. Ceiling keeps a hard
            // outer edge exactly at the radius.
            if( globalUniform.smokeStylize > 0.0 && globalUniform.smokeStylizeSteps > 0u )
            {
                const float n = float( globalUniform.smokeStylizeSteps );
                w = mix( w, ceil( w * n ) / n, globalUniform.smokeStylize );
            }

            const float d = w * globalUniform.smokeAlbedoDensity[ i ].a;

            acc += vec4( d * globalUniform.smokeAlbedoDensity[ i ].rgb, d );
        }
    }

    return acc;
}

// Combine the fog's medium with the smoke's into the single scattering albedo
// the froxel stores, weighted by their densities: a puff standing in fog is its
// OWN colour, and the air around it is still the fog's.
//
// THE COLLAPSE, and why it is an early-out rather than an identity. With
// smoke = vec4( 0 ) the general expression reduces to
//     ( fogDensity * fogTint + 0 ) / fogDensity
// which is fogTint ALGEBRAICALLY but not necessarily bit-for-bit: a multiply
// followed by a divide by the same value can land an ulp away. The fog ships
// with a measured transmittance ladder and ab-smoke.cmd fogsafe asserts a
// PIXEL-identical frame, so "close enough" is not the property wanted here --
// hence the early return, which is exact.
//
// The second guard covers a froxel with no medium at all: the old code reached
// that as `lighting * 0.0` and stored vec4( 0 ) anyway.
vec3 smoke_blendTint( vec3 fogTint, float fogDensity, vec4 smoke )
{
    if( smoke.a <= 0.0 )
    {
        return fogTint;
    }

    const float total = fogDensity + smoke.a;
    if( total <= 0.0 )
    {
        return fogTint;
    }

    const vec3 weighted = ( fogDensity * fogTint + smoke.rgb ) / total;

    // THE WEIGHTED AVERAGE IS HONEST AND LOSES THE SMOKE. smoke.rgb is the
    // albedo premultiplied by density, so a THIN puff contributes little and a
    // cell containing it comes out mostly the medium's colour -- which means
    // powder smoke goes beige in a beige room, blue in a blue one, and stops
    // reading as smoke at all. It was reported as "the gun smoke takes the room
    // colour and I can only see it with the flashlight", while a barrel's fat
    // burst, which carries enough density to win the average, stayed grey.
    //
    // The bias lets a smoke cell assert its own albedo beyond its density
    // share. smoke.rgb / smoke.a recovers the un-premultiplied colour, i.e.
    // what the puff would look like if it owned the cell outright.
    if( globalUniform.smokeTintBias > 0.0 && smoke.a > 0.0 )
    {
        const vec3 own = smoke.rgb / smoke.a;
        return mix( weighted, own, globalUniform.smokeTintBias );
    }

    return weighted;
}

#endif // SMOKE_H_
