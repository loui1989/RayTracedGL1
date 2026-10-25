# Copyright (c) 2026 RTGL1 Metal backend contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Compiles RTGL1's GLSL shaders (Source/Shaders) for the Metal backend:
#   GLSL --glslang--> SPIR-V --SpirvToMsl (SPIRV-Cross)--> MSL --metal--> .air --metallib--> <output>/<name>.metallib
# (a metallib per shader, as SPIRV-Cross emits the same helper functions in each)
# Raygen shaders become compute shaders that trace rays with ray queries (RAY_QUERY in RaygenCommon.h).
# Descriptor sets become Metal argument buffers, at buffer index = set index;
# the host fills them by member names (see Tools/SpirvToMsl.cpp).

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import MslIntersector

def entry_name( filename ):
    base, ext = os.path.splitext( filename )
    # vertex and fragment shaders might have the same base name (RsDecal.vert, RsDecal.frag)
    return base + "_vert" if ext == ".vert" else base + "_frag" if ext == ".frag" else base

# shader file : entry point name in the metallib
SHADERS = {
    os.path.basename( f ): entry_name( os.path.basename( f ) )
    for f in sorted( os.listdir( os.path.join( os.path.dirname( __file__ ), "../Shaders" ) ) )
    if f.endswith( ( ".rgen", ".comp", ".vert", ".frag" ) )
}

STAGES = { ".rgen": "comp", ".comp": "comp", ".vert": "vert", ".frag": "frag" }

THREADGROUP_SIZE = ( 8, 8, 1 )

DEFINES = [
    "METAL",
    "RAY_QUERY",
    "RAY_QUERY_GROUPED_INSTANCES",
]

def find_tool( name, override ):
    if override:
        return override
    # Homebrew's are newer than the ones that a Vulkan SDK may have put to /usr/local/bin
    for p in ( "/opt/homebrew/bin/" + name, shutil.which( name ) ):
        if p and os.path.exists( p ):
            return p
    sys.exit( "Can't find " + name )


class ToolError( Exception ):
    pass


def run( args ):
    r = subprocess.run( args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True )
    if r.returncode != 0:
        raise ToolError( r.stdout )
    return r.stdout


def as_metal_source( path ):
    # glslc enables #include by default, glslangValidator doesn't
    with open( path, "r" ) as f:
        src = f.read()
    lines = [
        "#version 460",
        "#extension GL_GOOGLE_include_directive : require",
    ]
    # a raygen shader as a compute shader: rays are traced with ray queries
    if path.endswith( ".rgen" ):
        lines += [
            "layout( local_size_x = %d, local_size_y = %d, local_size_z = %d ) in;" % THREADGROUP_SIZE,
            "#define gl_LaunchIDEXT gl_GlobalInvocationID",
        ]
    header = "\n".join( lines + [ "#line 1" ] ) + "\n"
    assert "#version 460" in src
    return header + src.replace( "#version 460", "", 1 )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--shaders", required=True, help="Source/Shaders folder" )
    parser.add_argument( "--generated", required=True, help="Source/Generated folder" )
    parser.add_argument( "--build", required=True, help="folder for intermediate files" )
    parser.add_argument( "--output", required=True, help="output folder for .metallib-s" )
    parser.add_argument( "--spirv-to-msl", required=True, help="Tools/SpirvToMsl executable" )
    parser.add_argument( "--glslang" )
    # without Xcode's Metal toolchain: the MSL sources are put to the output folder,
    # and compiled at run time (MetalComputePipelines)
    parser.add_argument( "--msl-only", action="store_true" )
    parser.add_argument( "--only", nargs="*", help="compile only these shader files" )
    args = parser.parse_args()

    glslang = find_tool( "glslangValidator", args.glslang )

    os.makedirs( args.build, exist_ok=True )
    os.makedirs( args.output, exist_ok=True )

    mslOnly = args.msl_only
    if not mslOnly:
        try:
            run( [ "xcrun", "-sdk", "macosx", "metal", "--version" ] )
        except ( ToolError, OSError ) as e:
            print( "Note: Xcode's Metal toolchain is not available, shaders will be compiled at run time "
                   "(to install it: xcodebuild -downloadComponent MetalToolchain)" )
            mslOnly = True

    failed = []
    for filename, entry in SHADERS.items():
      if args.only and filename not in args.only:
          continue
      try:
        src = os.path.join( args.shaders, filename )

        stage = STAGES[ os.path.splitext( filename )[ 1 ] ]
        comp  = os.path.join( args.build, entry + "." + stage )
        with open( comp, "w" ) as f:
            f.write( as_metal_source( src ) )

        spv = os.path.join( args.build, entry + ".spv" )
        msl = os.path.join( args.build, entry + ".metal" )
        air = os.path.join( args.build, entry + ".air" )

        run( [ glslang, "-V", "--target-env", "vulkan1.2", "-S", stage,
               "-I" + args.shaders, "-I" + args.generated ] +
             [ "-D" + d for d in DEFINES ] +
             [ comp, "-o", spv ] )

        threads = os.path.join( args.output, entry + ".threads" ) if stage == "comp" else "-"
        run( [ args.spirv_to_msl, spv, msl, entry, threads ] )

        # the intersector version of a ray tracing shader (MslIntersector.py), <entry>.isect.*:
        # used instead, if it compiles at run time / here
        with open( msl, "r" ) as f:
            isectSrc = MslIntersector.rewrite_for_intersector( f.read() )
        isect = os.path.join( args.build, entry + ".isect.metal" )
        for stale in ( entry + ".isect.metal", entry + ".isect.metallib" ):
            if os.path.exists( os.path.join( args.output, stale ) ):
                os.remove( os.path.join( args.output, stale ) )
        if isectSrc:
            with open( isect, "w" ) as f:
                f.write( isectSrc )

        if mslOnly:
            shutil.copyfile( msl, os.path.join( args.output, entry + ".metal" ) )
            if isectSrc:
                shutil.copyfile( isect, os.path.join( args.output, entry + ".isect.metal" ) )
            stale = os.path.join( args.output, entry + ".metallib" )
            if os.path.exists( stale ):
                os.remove( stale )
            print( "Translated " + filename + ( " (+ intersector version)" if isectSrc else "" ) )
        else:
            run( [ "xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O3", "-c", msl, "-o", air ] )
            run( [ "xcrun", "-sdk", "macosx", "metallib", air, "-o", os.path.join( args.output, entry + ".metallib" ) ] )
            if isectSrc:
                try:
                    isectAir = os.path.join( args.build, entry + ".isect.air" )
                    run( [ "xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O3", "-c", isect, "-o", isectAir ] )
                    run( [ "xcrun", "-sdk", "macosx", "metallib", isectAir, "-o",
                           os.path.join( args.output, entry + ".isect.metallib" ) ] )
                except ToolError as e:
                    print( "Note: the intersector version of " + filename + " didn't compile, the ray query one is used\n" +
                           "\n".join( str( e ).splitlines()[ :6 ] ) )
            print( "Compiled " + filename )
      except ToolError as e:
        failed.append( filename )
        print( "FAILED " + filename + "\n" + "\n".join( str( e ).splitlines()[ :12 ] ) )

    if failed:
        sys.exit( "%d shader(s) failed: %s" % ( len( failed ), " ".join( failed ) ) )


if __name__ == "__main__":
    main()
