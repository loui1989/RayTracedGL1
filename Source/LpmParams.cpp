// Copyright (c) 2020-2021 Sultim Tsyrendashiev
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

// Moved from ImageComposition.cpp, to share with the Metal backend

#include "LpmParams.h"

#include <cassert>
#include <cstdint>
// for ffx_a.h's CPU path
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define A_CPU 1
#include "Shaders/LPM/ffx_a.h"

namespace
{
void LpmSetupOut( AU1 i, inAU4 v )
{
    // this function version must not be called, only one with pContext
    assert( 0 );
}

void LpmSetupOut( void* pContext, AU1 i, const inAU4 value )
{
    AU1* ctl = static_cast< AU1* >( pContext );

    assert( i < 24 );

    ctl[ i * 4 + 0 ] = value[ 0 ];
    ctl[ i * 4 + 1 ] = value[ 1 ];
    ctl[ i * 4 + 2 ] = value[ 2 ];
    ctl[ i * 4 + 3 ] = value[ 3 ];
}
}

#include "Shaders/LPM/ffx_lpm.h"
#include "Shaders/LPM/LpmSetupCustom.inl"

void RTGL1::MakeLpmParams( void* pContext, const RgDrawFrameTonemappingParams& params, bool hdr )
{
#define LPM_RG_CONTEXT pContext,

    if( !hdr )
    {
        float contrast = params.contrast;

        varAF3( saturation ) = initAF3( params.saturation.data[ 0 ], //
                                        params.saturation.data[ 1 ],
                                        params.saturation.data[ 2 ] );

        varAF3( crosstalk ) = initAF3( params.crosstalk.data[ 0 ], //
                                       params.crosstalk.data[ 1 ],
                                       params.crosstalk.data[ 2 ] );

        LpmSetup( LPM_RG_CONTEXT false,
                  LPM_CONFIG_709_709,
                  LPM_COLORS_709_709,
                  0.0f,   // softGap
                  256.0f, // hdrMax
                  8.0f,   // exposure
                  contrast,
                  1.1f,   // shoulder contrast
                  saturation,
                  crosstalk );
    }
    else
    {
        float contrast = params.hdrContrast;

        varAF3( saturation ) = initAF3( params.hdrSaturation.data[ 0 ], //
                                        params.hdrSaturation.data[ 1 ],
                                        params.hdrSaturation.data[ 2 ] );

        varAF3( crosstalk ) = initAF3( params.crosstalk.data[ 0 ], //
                                       params.crosstalk.data[ 1 ],
                                       params.crosstalk.data[ 2 ] );

        const float hdr10S = LpmHdr10RawScalar( 1000 );

        LpmSetup( LPM_RG_CONTEXT false,
                  LPM_CONFIG_HDR10RAW_2020,
                  LPM_COLORS_HDR10RAW_2020,
                  0.0f,   // softGap
                  256.0f, // hdrMax
                  8.0f,   // exposure
                  contrast,
                  1.1f, // shoulder contrast
                  saturation,
                  crosstalk );
    }
}
