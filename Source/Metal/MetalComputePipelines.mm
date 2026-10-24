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

#include "MetalComputePipelines.h"

#include "RgException.h"

#include "Generated/ShaderCommonC.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace
{

// Tier 2 argument buffers: each resource is an 8 byte handle
constexpr NSUInteger HandleSize = sizeof( uint64_t );

constexpr NSUInteger UploadPageSize  = 4 * 1024 * 1024;
constexpr NSUInteger UploadAlignment = 64;

void WriteHandle( uint8_t* dst, NSUInteger offset, uint64_t handle )
{
    memcpy( dst + offset, &handle, sizeof( handle ) );
}

uint64_t ToHandle( MTLResourceID id )
{
    uint64_t h;
    static_assert( sizeof( id ) == sizeof( h ) );
    memcpy( &h, &id, sizeof( h ) );
    return h;
}

MTLResourceUsage ToUsage( MTLBindingAccess access )
{
    switch( access )
    {
        case MTLBindingAccessReadOnly: return MTLResourceUsageRead;
        case MTLBindingAccessWriteOnly: return MTLResourceUsageWrite;
        case MTLBindingAccessReadWrite:
        default: return MTLResourceUsageRead | MTLResourceUsageWrite;
    }
}

std::string MakeKey( const char* shaderName, const RTGL1::MetalComputePipelines::Constants& constants )
{
    std::string key = shaderName;
    for( auto [ id, value ] : constants )
    {
        key += ":" + std::to_string( id ) + "=" + std::to_string( value );
    }
    return key;
}

}



void RTGL1::MetalBindingTable::SetTexture( const std::string& name, id< MTLTexture > texture )
{
    entries[ name ] = Entry{ .resource = texture };
}

void RTGL1::MetalBindingTable::SetSampler( const std::string& name, id< MTLSamplerState > sampler )
{
    entries[ name ] = Entry{ .sampler = sampler };
}

void RTGL1::MetalBindingTable::SetBuffer( const std::string& name, id< MTLBuffer > buffer, NSUInteger offset )
{
    entries[ name ] = Entry{ .resource = buffer, .offset = offset };
}

void RTGL1::MetalBindingTable::SetAccelerationStructure(
    const std::string& name, id< MTLAccelerationStructure > tlas, std::span< const id< MTLAccelerationStructure > > blases )
{
    entries[ name ] = Entry{
        .resource       = tlas,
        .extraResources = { blases.begin(), blases.end() },
    };
}

void RTGL1::MetalBindingTable::SetArgumentBuffer( const std::string&                   name,
                                                  id< MTLBuffer >                      buffer,
                                                  std::span< const id< MTLResource > > resources )
{
    entries[ name ] = Entry{
        .resource       = buffer,
        .extraResources = { resources.begin(), resources.end() },
    };
}

const RTGL1::MetalBindingTable::Entry* RTGL1::MetalBindingTable::Find( const std::string& name ) const
{
    auto f = entries.find( name );
    return f != entries.end() ? &f->second : nullptr;
}



RTGL1::MetalComputePipelines::MetalComputePipelines( id< MTLDevice > _device, std::filesystem::path _folder )
    : device{ _device }, folder{ std::move( _folder ) }
{
    dummyBuffer       = [device newBufferWithLength:64 * 1024 options:MTLResourceStorageModeShared];
    dummyBuffer.label = @"Dummy buffer";
    memset( dummyBuffer.contents, 0, dummyBuffer.length );

    MTLSamplerDescriptor* s     = [MTLSamplerDescriptor new];
    s.supportArgumentBuffers    = YES;
    dummySampler                = [device newSamplerStateWithDescriptor:s];
}

void RTGL1::MetalComputePipelines::BeginFrame( uint32_t _frameIndex )
{
    frameIndex = _frameIndex;
    for( auto& page : uploadPages[ frameIndex ] )
    {
        page.used = 0;
    }
}

RTGL1::MetalComputePipelines::Upload RTGL1::MetalComputePipelines::AllocUpload( NSUInteger size )
{
    size = std::max< NSUInteger >( size, HandleSize );

    for( auto& page : uploadPages[ frameIndex ] )
    {
        NSUInteger offset = ( page.used + UploadAlignment - 1 ) / UploadAlignment * UploadAlignment;
        if( offset + size <= page.buffer.length )
        {
            page.used = offset + size;
            return Upload{ page.buffer, offset, static_cast< uint8_t* >( page.buffer.contents ) + offset };
        }
    }

    id< MTLBuffer > b = [device newBufferWithLength:std::max( UploadPageSize, size )
                                            options:MTLResourceStorageModeShared];
    b.label = @"Argument buffers";
    uploadPages[ frameIndex ].push_back( UploadPage{ b, size } );
    return Upload{ b, 0, static_cast< uint8_t* >( b.contents ) };
}

id< MTLFunction > RTGL1::MetalComputePipelines::GetFunction( const char* shaderName, const Constants& constants )
{
    NSError* error = nil;

    id< MTLLibrary > lib = libraries[ shaderName ];
    if( !lib )
    {
        const auto path = folder / ( std::string{ shaderName } + ".metallib" );
        lib             = [device newLibraryWithURL:[NSURL fileURLWithPath:@( path.c_str() )] error:&error];
        if( !lib )
        {
            throw RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                               "Can't load " + path.string() + ": " + error.localizedDescription.UTF8String );
        }
        libraries[ shaderName ] = lib;
    }

    // the declared types of the function constants: a value is the 32 bits of it (e.g. std::bit_cast of a float)
    std::map< uint32_t, MTLDataType > types;
    if( !constants.empty() )
    {
        if( id< MTLFunction > base = [lib newFunctionWithName:@( shaderName )] )
        {
            for( NSString* key in base.functionConstantsDictionary )
            {
                MTLFunctionConstant* c                 = base.functionConstantsDictionary[ key ];
                types[ uint32_t( c.index ) ] = c.type;
            }
        }
    }

    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for( auto [ id, value ] : constants )
    {
        const auto t = types.find( id );
        switch( t != types.end() ? t->second : MTLDataTypeUInt )
        {
            case MTLDataTypeFloat: {
                float f;
                static_assert( sizeof( f ) == sizeof( value ) );
                memcpy( &f, &value, sizeof( f ) );
                [values setConstantValue:&f type:MTLDataTypeFloat atIndex:id];
                break;
            }
            case MTLDataTypeBool: {
                const bool b = value != 0;
                [values setConstantValue:&b type:MTLDataTypeBool atIndex:id];
                break;
            }
            case MTLDataTypeInt: {
                const int32_t i = int32_t( value );
                [values setConstantValue:&i type:MTLDataTypeInt atIndex:id];
                break;
            }
            default: [values setConstantValue:&value type:MTLDataTypeUInt atIndex:id]; break;
        }
    }

    id< MTLFunction > fn = [lib newFunctionWithName:@( shaderName ) constantValues:values error:&error];
    if( !fn )
    {
        throw RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                           std::string{ "Can't find " } + shaderName + ": " + error.localizedDescription.UTF8String );
    }
    return fn;
}

void RTGL1::MetalComputePipelines::ParseBindings( NSArray< id< MTLBinding > >* bindings, StageBindings& dst )
{
    for( id< MTLBinding > b in bindings )
    {
        if( b.type != MTLBindingTypeBuffer )
        {
            continue;
        }
        auto*          bb = static_cast< id< MTLBufferBinding > >( b );
        MTLStructType* st = bb.bufferStructType;
        if( !st )
        {
            continue;
        }

        // SPIRV-Cross names descriptor sets' argument buffers spvDescriptorSet<N>,
        // any other buffer argument is the push_constant block
        if( ![b.name hasPrefix:@"spvDescriptorSet"] )
        {
            dst.pushConstantsIndex = b.index;
            dst.pushConstantsSize  = bb.bufferDataSize;
            continue;
        }

        ArgumentSet set{ .bufferIndex = b.index, .size = bb.bufferDataSize };
        for( MTLStructMember* m in st.members )
        {
            Member mem{
                .name            = m.name.UTF8String,
                .offset          = m.offset,
                .type            = m.dataType,
                .access          = MTLBindingAccessReadOnly,
                .textureDataType = MTLDataTypeNone,
                .textureType     = MTLTextureType2D,
                .isArray         = m.dataType == MTLDataTypeArray,
                .elementType     = MTLDataTypeNone,
            };
            if( auto t = m.textureReferenceType )
            {
                mem.access          = t.access;
                mem.textureDataType = t.textureDataType;
                mem.textureType     = t.textureType;
            }
            if( auto ptr = m.pointerType )
            {
                mem.access = ptr.access;
            }
            if( auto arr = m.arrayType )
            {
                mem.elementType = arr.elementType;
                if( auto t = arr.elementTextureReferenceType )
                {
                    mem.access          = t.access;
                    mem.textureDataType = t.textureDataType;
                    mem.textureType     = t.textureType;
                }
            }
            set.members.push_back( mem );
        }
        dst.sets.push_back( std::move( set ) );
    }
}

const RTGL1::MetalComputePipelines::Pipeline& RTGL1::MetalComputePipelines::Get( const char*      shaderName,
                                                                                 const Constants& constants )
{
    const std::string key = MakeKey( shaderName, constants );
    if( auto found = pipelines.find( key ); found != pipelines.end() )
    {
        return found->second;
    }

    NSError* error = nil;

    id< MTLFunction > fn = GetFunction( shaderName, constants );

    MTLComputePipelineReflection* reflection = nil;
    id< MTLComputePipelineState > pso =
        [device newComputePipelineStateWithFunction:fn
                                            options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                                         reflection:&reflection
                                              error:&error];
    if( !pso )
    {
        throw RgException( RG_RESULT_GRAPHICS_API_ERROR,
                           std::string{ "Can't create pipeline " } + shaderName + ": " +
                               error.localizedDescription.UTF8String );
    }

    Pipeline p{ .pso = pso };

    // written by CompileShadersMetal.py: the shader's local_size
    {
        uint32_t x = 1, y = 1, z = 1;
        std::ifstream f( folder / ( std::string{ shaderName } + ".threads" ) );
        if( !( f >> x >> y >> z ) )
        {
            throw RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                               std::string{ "Can't read the threadgroup size of " } + shaderName );
        }
        p.threadgroupSize = MTLSizeMake( x, y, z );
    }

    {
        StageBindings sb;
        ParseBindings( reflection.bindings, sb );
        p.sets               = std::move( sb.sets );
        p.pushConstantsIndex = sb.pushConstantsIndex;
        p.pushConstantsSize  = sb.pushConstantsSize;
    }

    return pipelines.emplace( key, std::move( p ) ).first->second;
}

id< MTLResource > RTGL1::MetalComputePipelines::GetDummy( const Member& m )
{
    if( m.type == MTLDataTypePointer )
    {
        return dummyBuffer;
    }

    const auto key = std::pair{ m.textureType, m.textureDataType };
    if( auto found = dummyTextures.find( key ); found != dummyTextures.end() )
    {
        return found->second;
    }

    MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
    desc.textureType           = m.textureType;
    desc.pixelFormat           = m.textureDataType == MTLDataTypeUInt  ? MTLPixelFormatR32Uint
                                 : m.textureDataType == MTLDataTypeInt ? MTLPixelFormatR32Sint
                                                                       : MTLPixelFormatRGBA16Float;
    desc.width       = 1;
    desc.height      = 1;
    desc.depth       = 1;
    desc.arrayLength = 1;
    desc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;

    id< MTLTexture > t = [device newTextureWithDescriptor:desc];
    t.label            = @"Dummy texture";
    dummyTextures[ key ] = t;
    return t;
}

template< typename SetBufferFunc >
void RTGL1::MetalComputePipelines::FillArgumentBuffers( const std::vector< ArgumentSet >& sets,
                                                        const MetalBindingTable&          bindings,
                                                        SetBufferFunc&&                   setBuffer,
                                                        std::vector< id< MTLResource > > ( &used )[ 3 ] )
{
    auto l_use = [ &used ]( id< MTLResource > r, MTLResourceUsage usage ) {
        if( r )
        {
            used[ usage == MTLResourceUsageRead ? 0 : usage == MTLResourceUsageWrite ? 1 : 2 ].push_back( r );
        }
    };

    auto l_missing = [ this ]( const Member& m ) {
        if( std::ranges::find( reportedMissing, m.name ) == reportedMissing.end() )
        {
            reportedMissing.push_back( m.name );
            debug::Verbose( "Metal backend: no resource for '{}', using a placeholder", m.name );
        }
    };

    for( const ArgumentSet& set : sets )
    {
        // a bindless array is its own argument buffer
        if( set.members.size() == 1 && set.members[ 0 ].isArray )
        {
            if( const auto* e = bindings.Find( set.members[ 0 ].name ) )
            {
                setBuffer( set.bufferIndex, static_cast< id< MTLBuffer > >( e->resource ), NSUInteger{ 0 } );
                for( id< MTLResource > r : e->extraResources )
                {
                    l_use( r, MTLResourceUsageRead );
                }
                continue;
            }
        }

        Upload up = AllocUpload( set.size );
        memset( up.ptr, 0, std::max< NSUInteger >( set.size, HandleSize ) );

        for( const Member& m : set.members )
        {
            const auto* e = bindings.Find( m.name );

            switch( m.type )
            {
                case MTLDataTypeTexture: {
                    id< MTLTexture > t = e ? static_cast< id< MTLTexture > >( e->resource ) : nil;
                    if( !t )
                    {
                        l_missing( m );
                        t = static_cast< id< MTLTexture > >( GetDummy( m ) );
                    }
                    WriteHandle( up.ptr, m.offset, ToHandle( t.gpuResourceID ) );
                    l_use( t, ToUsage( m.access ) );
                    break;
                }
                case MTLDataTypeSampler: {
                    id< MTLSamplerState > s = e && e->sampler ? e->sampler : dummySampler;
                    if( !e || !e->sampler )
                    {
                        l_missing( m );
                    }
                    WriteHandle( up.ptr, m.offset, ToHandle( s.gpuResourceID ) );
                    break;
                }
                case MTLDataTypePointer: {
                    id< MTLBuffer > b      = e ? static_cast< id< MTLBuffer > >( e->resource ) : nil;
                    NSUInteger      offset = e ? e->offset : 0;
                    if( !b )
                    {
                        l_missing( m );
                        b      = dummyBuffer;
                        offset = 0;
                    }
                    WriteHandle( up.ptr, m.offset, b.gpuAddress + offset );
                    l_use( b, ToUsage( m.access ) );
                    break;
                }
                case MTLDataTypeInstanceAccelerationStructure:
                case MTLDataTypePrimitiveAccelerationStructure: {
                    if( !e || !e->resource )
                    {
                        throw RgException( RG_RESULT_INTERNAL_ERROR,
                                           "Metal backend: no acceleration structure for " + m.name );
                    }
                    auto as = static_cast< id< MTLAccelerationStructure > >( e->resource );
                    WriteHandle( up.ptr, m.offset, ToHandle( as.gpuResourceID ) );
                    l_use( as, MTLResourceUsageRead );
                    for( id< MTLResource > r : e->extraResources )
                    {
                        l_use( r, MTLResourceUsageRead );
                    }
                    break;
                }
                case MTLDataTypeArray: {
                    // a bindless array without a registered argument buffer: a single placeholder
                    l_missing( m );
                    if( m.elementType == MTLDataTypeSampler )
                    {
                        WriteHandle( up.ptr, m.offset, ToHandle( dummySampler.gpuResourceID ) );
                    }
                    else
                    {
                        auto t = static_cast< id< MTLTexture > >( GetDummy( m ) );
                        WriteHandle( up.ptr, m.offset, ToHandle( t.gpuResourceID ) );
                        l_use( t, MTLResourceUsageRead );
                    }
                    break;
                }
                default:
                    throw RgException( RG_RESULT_INTERNAL_ERROR,
                                       "Metal backend: unsupported argument buffer member type for " + m.name );
            }
        }

        setBuffer( set.bufferIndex, up.buffer, up.offset );
    }

}

namespace
{
constexpr MTLResourceUsage Usages[] = {
    MTLResourceUsageRead,
    MTLResourceUsageWrite,
    MTLResourceUsageRead | MTLResourceUsageWrite,
};

std::vector< uint8_t > PadPushConstants( std::span< const uint8_t > src, NSUInteger size )
{
    std::vector< uint8_t > data( std::max< NSUInteger >( size, 16 ), 0 );
    memcpy( data.data(), src.data(), std::min( src.size(), data.size() ) );
    return data;
}
}

void RTGL1::MetalComputePipelines::Bind( id< MTLComputeCommandEncoder > enc,
                                         const Pipeline&                p,
                                         const MetalBindingTable&       bindings,
                                         std::span< const uint8_t >     pushConstants )
{
    if( p.pushConstantsIndex )
    {
        auto data = PadPushConstants( pushConstants, p.pushConstantsSize );
        [enc setBytes:data.data() length:data.size() atIndex:*p.pushConstantsIndex];
    }

    std::vector< id< MTLResource > > used[ 3 ];
    FillArgumentBuffers(
        p.sets,
        bindings,
        [ enc ]( NSUInteger index, id< MTLBuffer > buffer, NSUInteger offset ) {
            [enc setBuffer:buffer offset:offset atIndex:index];
        },
        used );

    for( int i = 0; i < 3; i++ )
    {
        if( !used[ i ].empty() )
        {
            [enc useResources:used[ i ].data() count:used[ i ].size() usage:Usages[ i ]];
        }
    }
}

namespace
{
std::string MakeRenderKey( const RTGL1::MetalComputePipelines::RenderDesc& d )
{
    std::string key = MakeKey( d.vertexShader, d.vertexConstants ) + "|" + MakeKey( d.fragmentShader, d.fragmentConstants );
    for( int i = 0; i < 4; i++ )
    {
        key += "|" + std::to_string( d.colorFormats[ i ] ) + ":" + std::to_string( d.blends[ i ].enable ) + ":" +
               std::to_string( d.blends[ i ].src ) + ":" + std::to_string( d.blends[ i ].dst ) + ":" +
               std::to_string( d.blends[ i ].separateAlpha ) + ":" + std::to_string( d.blends[ i ].srcAlpha ) + ":" +
               std::to_string( d.blends[ i ].dstAlpha );
    }
    key += "|" + std::to_string( d.depthFormat ) + "|" + std::to_string( d.rasterVertexLayout ) + "|" +
           std::to_string( d.lines );
    return key;
}
}

const RTGL1::MetalComputePipelines::RenderPipeline& RTGL1::MetalComputePipelines::GetRender( const RenderDesc& d )
{
    const std::string key = MakeRenderKey( d );
    if( auto found = renderPipelines.find( key ); found != renderPipelines.end() )
    {
        return found->second;
    }

    auto* desc             = [MTLRenderPipelineDescriptor new];
    desc.label             = [NSString stringWithFormat:@"%s + %s", d.vertexShader, d.fragmentShader];
    desc.vertexFunction    = GetFunction( d.vertexShader, d.vertexConstants );
    desc.fragmentFunction  = GetFunction( d.fragmentShader, d.fragmentConstants );
    desc.depthAttachmentPixelFormat = d.depthFormat;
    desc.inputPrimitiveTopology =
        d.lines ? MTLPrimitiveTopologyClassLine : MTLPrimitiveTopologyClassTriangle;

    for( int i = 0; i < 4; i++ )
    {
        MTLRenderPipelineColorAttachmentDescriptor* c = desc.colorAttachments[ i ];
        c.pixelFormat                                 = d.colorFormats[ i ];
        if( d.colorFormats[ i ] != MTLPixelFormatInvalid && d.blends[ i ].enable )
        {
            c.blendingEnabled             = YES;
            c.sourceRGBBlendFactor        = d.blends[ i ].src;
            c.destinationRGBBlendFactor   = d.blends[ i ].dst;
            c.sourceAlphaBlendFactor      = d.blends[ i ].separateAlpha ? d.blends[ i ].srcAlpha : d.blends[ i ].src;
            c.destinationAlphaBlendFactor = d.blends[ i ].separateAlpha ? d.blends[ i ].dstAlpha : d.blends[ i ].dst;
            c.rgbBlendOperation           = MTLBlendOperationAdd;
            c.alphaBlendOperation         = MTLBlendOperationAdd;
        }
    }

    if( d.rasterVertexLayout )
    {
        // RasterizedDataCollector::GetVertexLayout
        MTLVertexDescriptor* v = [MTLVertexDescriptor vertexDescriptor];

        v.attributes[ 0 ].format      = MTLVertexFormatFloat3;
        v.attributes[ 0 ].offset      = offsetof( ShVertex, position );
        v.attributes[ 0 ].bufferIndex = RasterVertexBufferIndex;

        v.attributes[ 1 ].format      = MTLVertexFormatUChar4Normalized;
        v.attributes[ 1 ].offset      = offsetof( ShVertex, color );
        v.attributes[ 1 ].bufferIndex = RasterVertexBufferIndex;

        v.attributes[ 2 ].format      = MTLVertexFormatFloat2;
        v.attributes[ 2 ].offset      = offsetof( ShVertex, texCoord );
        v.attributes[ 2 ].bufferIndex = RasterVertexBufferIndex;

        v.layouts[ RasterVertexBufferIndex ].stride       = sizeof( ShVertex );
        v.layouts[ RasterVertexBufferIndex ].stepFunction = MTLVertexStepFunctionPerVertex;

        desc.vertexDescriptor = v;
    }

    NSError*                     error      = nil;
    MTLRenderPipelineReflection* reflection = nil;
    id< MTLRenderPipelineState > pso =
        [device newRenderPipelineStateWithDescriptor:desc
                                             options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                                          reflection:&reflection
                                               error:&error];
    if( !pso )
    {
        throw RgException( RG_RESULT_GRAPHICS_API_ERROR,
                           std::string{ "Can't create render pipeline " } + d.vertexShader + " + " +
                               d.fragmentShader + ": " + error.localizedDescription.UTF8String );
    }

    RenderPipeline p{ .pso = pso };
    ParseBindings( reflection.vertexBindings, p.vertex );
    ParseBindings( reflection.fragmentBindings, p.fragment );

    return renderPipelines.emplace( key, std::move( p ) ).first->second;
}

void RTGL1::MetalComputePipelines::BindRender( id< MTLRenderCommandEncoder > enc,
                                               const RenderDesc&             desc,
                                               const MetalBindingTable&      bindings,
                                               std::span< const uint8_t >    pushConstants )
{
    const RenderPipeline& p = GetRender( desc );
    [enc setRenderPipelineState:p.pso];

    if( p.vertex.pushConstantsIndex )
    {
        auto data = PadPushConstants( pushConstants, p.vertex.pushConstantsSize );
        [enc setVertexBytes:data.data() length:data.size() atIndex:*p.vertex.pushConstantsIndex];
    }
    if( p.fragment.pushConstantsIndex )
    {
        auto data = PadPushConstants( pushConstants, p.fragment.pushConstantsSize );
        [enc setFragmentBytes:data.data() length:data.size() atIndex:*p.fragment.pushConstantsIndex];
    }

    std::vector< id< MTLResource > > used[ 3 ];
    FillArgumentBuffers(
        p.vertex.sets,
        bindings,
        [ enc ]( NSUInteger index, id< MTLBuffer > buffer, NSUInteger offset ) {
            [enc setVertexBuffer:buffer offset:offset atIndex:index];
        },
        used );
    FillArgumentBuffers(
        p.fragment.sets,
        bindings,
        [ enc ]( NSUInteger index, id< MTLBuffer > buffer, NSUInteger offset ) {
            [enc setFragmentBuffer:buffer offset:offset atIndex:index];
        },
        used );

    for( int i = 0; i < 3; i++ )
    {
        if( !used[ i ].empty() )
        {
            [enc useResources:used[ i ].data()
                        count:used[ i ].size()
                        usage:Usages[ i ]
                       stages:MTLRenderStageVertex | MTLRenderStageFragment];
        }
    }
}

id< MTLDepthStencilState > RTGL1::MetalComputePipelines::GetDepthState( bool test, bool write )
{
    // must test, if writes
    test = test || write;

    const uint32_t key = ( test ? 1u : 0u ) | ( write ? 2u : 0u );
    if( auto found = depthStates.find( key ); found != depthStates.end() )
    {
        return found->second;
    }

    auto* d                 = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction  = test ? MTLCompareFunctionLessEqual : MTLCompareFunctionAlways;
    d.depthWriteEnabled     = write;

    id< MTLDepthStencilState > state = [device newDepthStencilStateWithDescriptor:d];
    depthStates[ key ]               = state;
    return state;
}

MTLSize RTGL1::MetalComputePipelines::GetThreadgroupSize( const char* shaderName, const Constants& constants )
{
    return Get( shaderName, constants ).threadgroupSize;
}

void RTGL1::MetalComputePipelines::Dispatch( id< MTLComputeCommandEncoder > enc,
                                             const char*                    shaderName,
                                             const Constants&               constants,
                                             const MetalBindingTable&       bindings,
                                             uint32_t                       threadsX,
                                             uint32_t                       threadsY,
                                             uint32_t                       threadsZ,
                                             std::span< const uint8_t >     pushConstants )
{
    const Pipeline& p = Get( shaderName, constants );
    [enc setComputePipelineState:p.pso];
    Bind( enc, p, bindings, pushConstants );
    // exactly as many threads as vkCmdTraceRaysKHR launches: raygen shaders don't check bounds
    [enc dispatchThreads:MTLSizeMake( threadsX, threadsY, threadsZ ) threadsPerThreadgroup:p.threadgroupSize];
}

void RTGL1::MetalComputePipelines::DispatchThreadgroups( id< MTLComputeCommandEncoder > enc,
                                                         const char*                    shaderName,
                                                         const Constants&               constants,
                                                         const MetalBindingTable&       bindings,
                                                         uint32_t                       groupsX,
                                                         uint32_t                       groupsY,
                                                         uint32_t                       groupsZ,
                                                         std::span< const uint8_t >     pushConstants )
{
    const Pipeline& p = Get( shaderName, constants );
    [enc setComputePipelineState:p.pso];
    Bind( enc, p, bindings, pushConstants );
    [enc dispatchThreadgroups:MTLSizeMake( groupsX, groupsY, groupsZ ) threadsPerThreadgroup:p.threadgroupSize];
}
