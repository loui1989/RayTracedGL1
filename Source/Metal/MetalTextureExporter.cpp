// Copyright (c) 2022 Sultim Tsyrendashiev
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

// The part of TextureExporter.cpp that doesn't depend on Vulkan,
// for rgUtilExportAsTGA in the Metal build.

#include "TextureExporter.h"
#include "Utils.h"

#include "Stb/stb_image_write.h"

namespace
{

constexpr uint64_t DstBytesPerPixel = 4;

}

bool RTGL1::TextureExporter::WriteTGA( std::filesystem::path filepath,
                                       const void*           pixels,
                                       const RgExtent2D&     size )
{
    assert( std::filesystem::exists( filepath.parent_path() ) );

    if( !stbi_write_tga( filepath.replace_extension( ".tga" ).string().c_str(),
                         int( size.width ),
                         int( size.height ),
                         DstBytesPerPixel,
                         pixels ) )
    {
        debug::Warning( "{}: stbi_write_tga fail", filepath.string() );
        return false;
    }

    return true;
}
