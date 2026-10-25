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

# The intersector version of a ray tracing shader (SPIRV-Cross MSL of RaygenCommon.h's traces):
# traceRayQuery / traceRayQueryAnyHit trace with Metal's intersector instead of an
# intersection_query loop, and the alpha test (rtAlphaTestAccepts) runs as an intersection
# function, called by the intersector for the non-opaque (alpha-tested) geometries only.
# On Apple GPUs, this is faster than the query loop.
#
# The kernel gets the intersection function table at INTERSECTION_FUNCTION_TABLE_INDEX
# (MetalComputePipelines binds it), the intersection function is INTERSECTION_FUNCTION.
# The resources of the alpha test are given to the intersection function in the ray payload.
#
# Returns None if the shader doesn't have the expected shape: then the query version is used.

import re

INTERSECTION_FUNCTION_TABLE_INDEX = 29
INTERSECTION_FUNCTION = "rtglAlphaTest"

QUERY_TYPE = "raytracing::intersection_query<raytracing::instancing, raytracing::triangle_data>"
TABLE_TYPE = "raytracing::intersection_function_table<raytracing::triangle_data, raytracing::instancing>"


class Func:
    def __init__( self, params_begin, params_end, body_begin, body_end ):
        self.params_begin = params_begin
        self.params_end   = params_end
        self.body_begin   = body_begin
        self.body_end     = body_end


def match_close( s, i, open_c, close_c ):
    """index of the bracket that closes the one at s[i]"""
    depth = 0
    for j in range( i, len( s ) ):
        if s[ j ] == open_c:
            depth += 1
        elif s[ j ] == close_c:
            depth -= 1
            if depth == 0:
                return j
    return -1


def find_function( msl, name ):
    """the definition of a function (SPIRV-Cross puts them at column 0)"""
    for m in re.finditer( r"^[A-Za-z_][^\n;{}()]*\b" + re.escape( name ) + r"\(", msl, re.MULTILINE ):
        p = m.end() - 1
        close = match_close( msl, p, "(", ")" )
        if close < 0:
            return None
        b = close + 1
        while b < len( msl ) and msl[ b ] in " \t\r\n":
            b += 1
        if b < len( msl ) and msl[ b ] == "{":
            e = match_close( msl, b, "{", "}" )
            if e < 0:
                return None
            return Func( p + 1, close, b, e + 1 )
    return None


def split_params( params ):
    out, depth, cur = [], 0, ""
    for c in params:
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        if c == "," and depth == 0:
            out.append( cur.strip() )
            cur = ""
            continue
        cur += c
    if cur.strip():
        out.append( cur.strip() )
    return out


def param_name( p ):
    m = re.search( r"([A-Za-z_][A-Za-z0-9_]*)\s*$", p )
    return m.group( 1 ) if m else ""


class Res:
    """a resource parameter of the alpha test, and how it's kept in the payload"""

    def __init__( self, param ):
        self.name = param_name( param )
        decl      = param[ : len( param ) - len( self.name ) ].strip()
        self.byRef = decl.endswith( "&" )
        if self.byRef:
            # "const device X_BT&" -> a pointer
            self.type = decl[ :-1 ].strip()
            if not ( " device " in " " + self.type + " " or " constant " in " " + self.type + " " ):
                raise ValueError( "not a device / constant reference: " + param )
            self.member = self.type + "* " + self.name
            self.store  = "&" + self.name
            self.load   = "*rtglPayload." + self.name
        else:
            m = re.match( r"^(const\s+)?spvDescriptorArray<(.+)>$", decl )
            if not m:
                raise ValueError( "a resource by value: " + param )
            # a bindless array: its pointer
            self.type   = "spvDescriptorArray<" + m.group( 2 ) + ">"
            self.member = "const device void* " + self.name
            self.store  = "( const device void* )" + self.name + ".ptr"
            self.load   = self.type + "( rtglPayload." + self.name + " )"


def make_trace_body( msl, f, is_anyhit, res ):
    ps = split_params( msl[ f.params_begin : f.params_end ] )
    if len( ps ) < ( 7 if is_anyhit else 6 ):
        raise ValueError( "trace function parameters" )
    asName = next( ( param_name( p ) for p in ps if "acceleration_structure" in p ), "" )
    rqName = next( ( param_name( p ) for p in ps if QUERY_TYPE in p ), "" )
    if not asName or not rqName:
        raise ValueError( "no acceleration structure / query parameter" )
    names = { param_name( p ) for p in ps }
    for r in res:
        if r.name not in names:
            raise ValueError( "trace function lacks " + r.name )
    flags, mask, origin, tmin, direction, tmax = ( param_name( p ) for p in ps[ :6 ] )
    media = param_name( ps[ 6 ] ) if is_anyhit else "false"

    b = "{\n"
    b += "    RtglAlphaPayload rtglPayload = { " + ", ".join( [ media ] + [ r.store for r in res ] ) + " };\n"
    b += "    raytracing::intersector<raytracing::triangle_data, raytracing::instancing> rtglIt;\n"
    # gl_RayFlagsTerminateOnFirstHitEXT for the any-hit trace
    b += "    rtglConfigure( rtglIt, " + flags + ( " | 4u" if is_anyhit else "" ) + " );\n"
    b += ( "    auto rtglHit = rtglIt.intersect( raytracing::ray( " + origin + ", " + direction + ", " + tmin + ", " +
           tmax + " ), " + asName + ", " + mask + " & 255u, " + rqName + ", rtglPayload );\n" )
    if is_anyhit:
        b += "    return rtglHit.type != raytracing::intersection_type::none;\n}"
    else:
        # as RaygenCommon.h's traceRayQuery with RAY_QUERY_GROUPED_INSTANCES:
        # instance = ( first geometry index << 4 | custom flags ), ShGeometryInstance = first + geometry index
        b += ( "    ShPayload p;\n"
               "    p.baryCoords = float2( 0.0 );\n"
               "    p.instIdAndIndex = 4294967295u;\n"
               "    p.geomAndPrimIndex = 4294967295u;\n"
               "    if( rtglHit.type == raytracing::intersection_type::triangle )\n"
               "    {\n"
               "        const uint rtglUser = uint( rtglHit.user_instance_id );\n"
               "        const uint rtglInst = ( rtglUser >> 4u ) + uint( rtglHit.geometry_id );\n"
               "        p.baryCoords = rtglHit.triangle_barycentric_coord;\n"
               "        p.instIdAndIndex = ( rtglInst << 4u ) | ( rtglUser & 15u );\n"
               "        p.geomAndPrimIndex = uint( rtglHit.primitive_id );\n"
               "    }\n"
               "    return p;\n}" )
    return b


def rewrite_for_intersector( msl ):
    try:
        return _rewrite( msl )
    except ValueError:
        return None


def _rewrite( msl ):
    alpha = find_function( msl, "rtAlphaTestAccepts" )
    cand  = find_function( msl, "rqIsCandidateAccepted" )
    if not alpha or not cand:
        return None
    closest = find_function( msl, "traceRayQuery" )
    anyhit  = find_function( msl, "traceRayQueryAnyHit" )
    if not closest and not anyhit:
        return None

    # the resources of the alpha test: after instanceId, primitiveId, b, media
    alphaParams = split_params( msl[ alpha.params_begin : alpha.params_end ] )
    if len( alphaParams ) < 4:
        raise ValueError( "alpha test parameters" )
    res = [ Res( p ) for p in alphaParams[ 4: ] ]

    edits = [ ( cand.body_begin, cand.body_end, "{\n    return true;\n}" ) ]
    if closest:
        edits.append( ( closest.body_begin, closest.body_end, make_trace_body( msl, closest, False, res ) ) )
    if anyhit:
        edits.append( ( anyhit.body_begin, anyhit.body_end, make_trace_body( msl, anyhit, True, res ) ) )
    for begin, end, text in sorted( edits, key=lambda e: -e[ 0 ] ):
        msl = msl[ :begin ] + text + msl[ end: ]

    # nothing else may use the query
    for use in ( ".next()", "get_candidate", "get_committed", ".reset(", ".commit_triangle_intersection" ):
        if use in msl:
            raise ValueError( "the query is used elsewhere: " + use )

    msl = msl.replace( QUERY_TYPE, "RtglTracer" )

    # the kernel: the function table as an argument, and the tracer initialized with it
    k = msl.find( "kernel void " )
    if k < 0:
        raise ValueError( "no kernel" )
    po = msl.find( "(", k )
    pc = match_close( msl, po, "(", ")" )
    if pc < 0:
        raise ValueError( "kernel parameters" )
    msl = ( msl[ :pc ] + ", " + TABLE_TYPE + " rtglFunctionTable [[buffer(" +
            str( INTERSECTION_FUNCTION_TABLE_INDEX ) + ")]]" + msl[ pc: ] )
    decl = re.search( r"^(\s+)RtglTracer (\w+);", msl[ k: ], re.MULTILINE )
    if not decl:
        raise ValueError( "no tracer in the kernel" )
    at = k + decl.start()
    msl = msl[ :at ] + decl.group( 1 ) + "RtglTracer " + decl.group( 2 ) + " = rtglFunctionTable;" + msl[ k + decl.end(): ]
    if len( re.findall( r"\bRtglTracer \w+;", msl ) ) != 0:
        raise ValueError( "another tracer variable" )

    # the types and the helper before the first function
    types = "struct RtglAlphaPayload\n{\n    bool media;\n"
    for r in res:
        types += "    " + r.member + ";\n"
    types += "};\n\n"
    types += "typedef " + TABLE_TYPE + " RtglTracer;\n\n"
    types += ( "// Vulkan ray flags -> intersector options (as spvMakeIntersectionParams)\n"
               "template< typename I >\n"
               "static inline __attribute__((always_inline)) void rtglConfigure( thread I& it, uint flags )\n"
               "{\n"
               "    it.assume_geometry_type( raytracing::geometry_type::triangle );\n"
               "    if( ( flags & 1u ) != 0u ) it.force_opacity( raytracing::forced_opacity::opaque );\n"
               "    if( ( flags & 2u ) != 0u ) it.force_opacity( raytracing::forced_opacity::non_opaque );\n"
               "    if( ( flags & 4u ) != 0u ) it.accept_any_intersection( true );\n"
               "    if( ( flags & 16u ) != 0u ) it.set_triangle_cull_mode( raytracing::triangle_cull_mode::back );\n"
               "    if( ( flags & 32u ) != 0u ) it.set_triangle_cull_mode( raytracing::triangle_cull_mode::front );\n"
               "    if( ( flags & 64u ) != 0u ) it.set_opacity_cull_mode( raytracing::opacity_cull_mode::opaque );\n"
               "    if( ( flags & 128u ) != 0u ) it.set_opacity_cull_mode( raytracing::opacity_cull_mode::non_opaque );\n"
               "}\n\n" )
    # right before the alpha test's definition (and its attribute line): after the types it uses,
    # before the traces
    line = msl.rfind( "\n", 0, find_function( msl, "rtAlphaTestAccepts" ).params_begin ) + 1
    attr = "static inline __attribute__((always_inline))\n"
    if msl[ max( 0, line - len( attr ) ) : line ] == attr:
        line -= len( attr )
    msl = msl[ :line ] + types + msl[ line: ]

    # the intersection function, last
    # (the tags in the attribute unqualified, as in Apple's samples)
    fn = ( "\nusing namespace metal::raytracing;\n\n[[intersection(triangle, triangle_data, instancing)]]\n"
           "bool " + INTERSECTION_FUNCTION + "( uint rtglUser [[user_instance_id]], uint rtglGeometry [[geometry_id]],\n"
           "    uint rtglPrimitive [[primitive_id]], float2 rtglBary [[barycentric_coord]],\n"
           "    ray_data RtglAlphaPayload& rtglPayload [[payload]] )\n"
           "{\n"
           "    const int    instanceId  = int( rtglUser >> 4u ) + int( rtglGeometry );\n"
           "    const uint   primitiveId = rtglPrimitive;\n"
           "    const float2 b           = rtglBary;\n"
           "    const bool   media       = rtglPayload.media;\n"
           "    return rtAlphaTestAccepts( instanceId, primitiveId, b, media" )
    for r in res:
        fn += ",\n        " + r.load
    fn += " );\n}\n"
    return msl + fn
