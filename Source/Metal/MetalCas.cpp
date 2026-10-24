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

// CasSetup of AMD FidelityFX CAS on CPU, for CmCas.comp (the same as Sharpening.cpp)

#include "MetalCas.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define A_CPU 1
#include "Shaders/CAS/ffx_a.h"
#include "Shaders/CAS/ffx_cas.h"

void RTGL1::MakeCasPush( uint32_t ( &con0 )[ 4 ],
                         uint32_t ( &con1 )[ 4 ],
                         float    sharpness,
                         uint32_t width,
                         uint32_t height )
{
    CasSetup( con0, con1, sharpness, float( width ), float( height ), float( width ), float( height ) );
}
