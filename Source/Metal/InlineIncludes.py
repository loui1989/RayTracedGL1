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

# Writes a Metal source with its #include "..." files inlined (each once), so that it can be
# compiled at run time with newLibraryWithSource, which has no include paths:
# for a build without Xcode's Metal toolchain.
#   InlineIncludes.py <input.metal> <output.metal> <include dir>...

import os
import re
import sys

def main():
    src, dst, dirs = sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 3: ]
    seen = set()

    def expand( path ):
        out = []
        with open( path, "r" ) as f:
            for line in f:
                m = re.match( r'\s*#\s*include\s+"([^"]+)"', line )
                if not m:
                    if re.match( r'\s*#\s*pragma\s+once', line ):
                        continue
                    out.append( line )
                    continue
                name = m.group( 1 )
                for d in [ os.path.dirname( path ) ] + dirs:
                    candidate = os.path.join( d, name )
                    if os.path.exists( candidate ):
                        break
                else:
                    sys.exit( "Can't find " + name + " included from " + path )
                key = os.path.realpath( candidate )
                if key not in seen:
                    seen.add( key )
                    out.append( "// begin " + name + "\n" )
                    out += expand( candidate )
                    out.append( "// end " + name + "\n" )
        return out

    text = "".join( expand( src ) )
    os.makedirs( os.path.dirname( os.path.abspath( dst ) ), exist_ok=True )
    with open( dst, "w" ) as f:
        f.write( text )

if __name__ == "__main__":
    main()
