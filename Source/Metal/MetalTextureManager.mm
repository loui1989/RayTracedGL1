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

#include "MetalTextureManager.h"

#include "Common.h"
#include "Const.h"
#include "DrawFrameInfo.h"
#include "LibraryConfig.h"
#include "RgException.h"
#include "Utils.h"

#include "Generated/ShaderCommonC.h"

#include <algorithm>
#include <cmath>
#include <ranges>

namespace
{

struct MetalFormat
{
    MTLPixelFormat format{ MTLPixelFormatInvalid };
    // for uncompressed formats: 1x1 block
    uint32_t       blockSize{ 1 };
    uint32_t       bytesPerBlock{ 0 };

    bool IsCompressed() const { return blockSize > 1; }
};

// The formats that ImageLoader (.ktx2), ImageLoaderDev and RgOriginalTextureInfo produce
MetalFormat ToMetalFormat( VkFormat f )
{
    switch( f )
    {
        case VK_FORMAT_R8_UNORM: return { MTLPixelFormatR8Unorm, 1, 1 };
        case VK_FORMAT_R8_SRGB: return { MTLPixelFormatR8Unorm_sRGB, 1, 1 };
        case VK_FORMAT_R8G8_UNORM: return { MTLPixelFormatRG8Unorm, 1, 2 };
        case VK_FORMAT_R8G8_SRGB: return { MTLPixelFormatRG8Unorm_sRGB, 1, 2 };
        case VK_FORMAT_R8G8B8A8_UNORM: return { MTLPixelFormatRGBA8Unorm, 1, 4 };
        case VK_FORMAT_R8G8B8A8_SRGB: return { MTLPixelFormatRGBA8Unorm_sRGB, 1, 4 };
        case VK_FORMAT_B8G8R8A8_UNORM: return { MTLPixelFormatBGRA8Unorm, 1, 4 };
        case VK_FORMAT_B8G8R8A8_SRGB: return { MTLPixelFormatBGRA8Unorm_sRGB, 1, 4 };
        case VK_FORMAT_R16G16B16A16_SFLOAT: return { MTLPixelFormatRGBA16Float, 1, 8 };
        case VK_FORMAT_R32G32B32A32_SFLOAT: return { MTLPixelFormatRGBA32Float, 1, 16 };

        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return { MTLPixelFormatBC1_RGBA, 4, 8 };
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return { MTLPixelFormatBC1_RGBA_sRGB, 4, 8 };
        case VK_FORMAT_BC2_UNORM_BLOCK: return { MTLPixelFormatBC2_RGBA, 4, 16 };
        case VK_FORMAT_BC2_SRGB_BLOCK: return { MTLPixelFormatBC2_RGBA_sRGB, 4, 16 };
        case VK_FORMAT_BC3_UNORM_BLOCK: return { MTLPixelFormatBC3_RGBA, 4, 16 };
        case VK_FORMAT_BC3_SRGB_BLOCK: return { MTLPixelFormatBC3_RGBA_sRGB, 4, 16 };
        case VK_FORMAT_BC4_UNORM_BLOCK: return { MTLPixelFormatBC4_RUnorm, 4, 8 };
        case VK_FORMAT_BC4_SNORM_BLOCK: return { MTLPixelFormatBC4_RSnorm, 4, 8 };
        case VK_FORMAT_BC5_UNORM_BLOCK: return { MTLPixelFormatBC5_RGUnorm, 4, 16 };
        case VK_FORMAT_BC5_SNORM_BLOCK: return { MTLPixelFormatBC5_RGSnorm, 4, 16 };
        case VK_FORMAT_BC6H_UFLOAT_BLOCK: return { MTLPixelFormatBC6H_RGBUfloat, 4, 16 };
        case VK_FORMAT_BC6H_SFLOAT_BLOCK: return { MTLPixelFormatBC6H_RGBFloat, 4, 16 };
        case VK_FORMAT_BC7_UNORM_BLOCK: return { MTLPixelFormatBC7_RGBAUnorm, 4, 16 };
        case VK_FORMAT_BC7_SRGB_BLOCK: return { MTLPixelFormatBC7_RGBAUnorm_sRGB, 4, 16 };

        default: return {};
    }
}

VkFormat ToVkFormat( RgFormat f )
{
    switch( f )
    {
        case RG_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
        case RG_FORMAT_R8_SRGB: return VK_FORMAT_R8_SRGB;
        case RG_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case RG_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case RG_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case RG_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case RG_FORMAT_UNDEFINED:
        default: return VK_FORMAT_R8G8B8A8_SRGB;
    }
}

// TextureManager.cpp's getVkFormat
VkFormat GetFormat( const RgOriginalTextureDetailsEXT* details, VkFormat fallback )
{
    if( details && details->format != RG_FORMAT_UNDEFINED )
    {
        return ToVkFormat( details->format );
    }
    return fallback;
}

// TextureUploader::CreateImageView's swizzling
MTLTextureSwizzleChannels ToMetalSwizzle( std::optional< RgTextureSwizzling > swizzling )
{
    constexpr auto R = MTLTextureSwizzleRed, G = MTLTextureSwizzleGreen, B = MTLTextureSwizzleBlue,
                   A = MTLTextureSwizzleAlpha, ONE = MTLTextureSwizzleOne;

    if( !swizzling )
    {
        return MTLTextureSwizzleChannelsMake( R, G, B, A );
    }
    switch( *swizzling )
    {
        case RG_TEXTURE_SWIZZLING_NULL_ROUGHNESS_METALLIC: return MTLTextureSwizzleChannelsMake( ONE, G, B, A );
        case RG_TEXTURE_SWIZZLING_NULL_METALLIC_ROUGHNESS: return MTLTextureSwizzleChannelsMake( ONE, B, G, A );
        case RG_TEXTURE_SWIZZLING_OCCLUSION_ROUGHNESS_METALLIC: return MTLTextureSwizzleChannelsMake( R, G, B, A );
        case RG_TEXTURE_SWIZZLING_OCCLUSION_METALLIC_ROUGHNESS: return MTLTextureSwizzleChannelsMake( R, B, G, A );
        case RG_TEXTURE_SWIZZLING_ROUGHNESS_METALLIC: return MTLTextureSwizzleChannelsMake( ONE, R, G, A );
        case RG_TEXTURE_SWIZZLING_METALLIC_ROUGHNESS: return MTLTextureSwizzleChannelsMake( ONE, G, R, A );
        default: assert( 0 ); return MTLTextureSwizzleChannelsMake( R, G, B, A );
    }
}

MTLSamplerAddressMode ToMetal( RgSamplerAddressMode m )
{
    return m == RG_SAMPLER_ADDRESS_MODE_CLAMP ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
}

constexpr RTGL1::MaterialTextures EmptyMaterialTextures = { {
    RTGL1::EMPTY_TEXTURE_INDEX,
    RTGL1::EMPTY_TEXTURE_INDEX,
    RTGL1::EMPTY_TEXTURE_INDEX,
    RTGL1::EMPTY_TEXTURE_INDEX,
    RTGL1::EMPTY_TEXTURE_INDEX,
} };

constexpr bool PreferExistingMaterials = true;

constexpr const char* Postfixes[] = {
    RTGL1::TEXTURE_ALBEDO_ALPHA_POSTFIX,
    RTGL1::TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_POSTFIX,
    RTGL1::TEXTURE_NORMAL_POSTFIX,
    RTGL1::TEXTURE_EMISSIVE_POSTFIX,
    RTGL1::TEXTURE_HEIGHT_POSTFIX,
};
static_assert( std::size( Postfixes ) == RTGL1::TEXTURES_PER_MATERIAL_COUNT );

}

RTGL1::MetalTextureManager::MetalTextureManager( id< MTLDevice >       _device,
                                                 id< MTLCommandQueue > _queue,
                                                 std::filesystem::path _ovrdFolder,
                                                 RgTextureSwizzling    _pbrSwizzling,
                                                 bool                  _forceMinificationFilterLinear,
                                                 bool                  _forceNormalMapFilterLinear )
    : device{ _device }
    , queue{ _queue }
    , ovrdFolder{ std::move( _ovrdFolder ) }
    , pbrSwizzling{ _pbrSwizzling }
    , forceMinificationFilterLinear{ _forceMinificationFilterLinear }
    , forceNormalMapFilterLinear{ _forceNormalMapFilterLinear }
    , isdevmode{ LibConfig().developerMode }
    , imageLoaderKtx{ std::make_unique< ImageLoader >() }
    , imageLoaderRaw{ std::make_unique< ImageLoaderDev >() }
{
    table = [device newBufferWithLength:sizeof( TableEntry ) * MaxTextureCount
                                options:MTLResourceStorageModeShared];
    table.label = @"Texture table";

    imageTable = [device newBufferWithLength:sizeof( MTLResourceID ) * MaxTextureCount
                                     options:MTLResourceStorageModeShared];
    imageTable.label = @"Texture image table";
    samplerTable = [device newBufferWithLength:sizeof( MTLResourceID ) * MaxTextureCount
                                       options:MTLResourceStorageModeShared];
    samplerTable.label = @"Texture sampler table";

    slots.reserve( MaxTextureCount );

    // TextureManager::CreateEmptyTexture: white, so that sampling it doesn't change a color
    {
        static_assert( EMPTY_TEXTURE_INDEX == 0 && MATERIAL_NO_TEXTURE == 0 );
        constexpr uint32_t white = 0xFFFFFFFF;

        const auto info = ImageLoader::ResultInfo{
            .levelOffsets   = { 0 },
            .levelSizes     = { sizeof( white ) },
            .levelCount     = 1,
            .isPregenerated = false,
            .pData          = reinterpret_cast< const uint8_t* >( &white ),
            .dataSize       = sizeof( white ),
            .baseSize       = { 1, 1 },
            .format         = VK_FORMAT_R8G8B8A8_UNORM,
        };
        const uint32_t index = PrepareTexture(
            info,
            SamplerManager::Handle{ RG_SAMPLER_FILTER_NEAREST, RG_SAMPLER_ADDRESS_MODE_REPEAT, RG_SAMPLER_ADDRESS_MODE_REPEAT },
            false,
            "Empty",
            std::nullopt,
            {} );
        if( index != EMPTY_TEXTURE_INDEX || slots.size() != 1 )
        {
            throw RgException( RG_RESULT_INTERNAL_ERROR, "Couldn't create the empty texture" );
        }
    }

    // the same files as VulkanDevice passes to TextureManager
    waterNormalTextureIndex   = CreateSingleTexture( ovrdFolder / "WaterNormal_n.ktx2", false, "Water normal" );
    dirtMaskTextureIndex      = CreateSingleTexture( ovrdFolder / "DirtMask.ktx2", true, "Dirt mask" );
    sceneBuildingTextureIndex = CreateSingleTexture( ovrdFolder / "SceneBuildWarning.ktx2", true, "Scene building" );
}

RTGL1::MetalTextureManager::~MetalTextureManager() = default;

RTGL1::TextureOverrides::Loader RTGL1::MetalTextureManager::OnlyKTX2LoaderIfNonDevMode()
{
    if( isdevmode )
    {
        return std::tuple{ imageLoaderRaw.get(), imageLoaderKtx.get() };
    }
    return std::tuple{ imageLoaderKtx.get() };
}

RTGL1::TextureOverrides::Loader RTGL1::MetalTextureManager::AnyImageLoader()
{
    // prefer raw formats in devmode
    if( isdevmode )
    {
        return std::tuple{ imageLoaderRaw.get(), imageLoaderKtx.get() };
    }
    return std::tuple{ imageLoaderKtx.get(), imageLoaderRaw.get() };
}

uint32_t RTGL1::MetalTextureManager::CreateSingleTexture( const std::filesystem::path& path,
                                                         bool                         isSRGB,
                                                         const char*                  debugName )
{
    if( !std::filesystem::is_regular_file( path ) )
    {
        return EMPTY_TEXTURE_INDEX;
    }
    auto ovrd = TextureOverrides{ path, isSRGB, std::tuple{ imageLoaderKtx.get() } };
    return PrepareTexture(
        ovrd.result,
        SamplerManager::Handle{ RG_SAMPLER_FILTER_LINEAR, RG_SAMPLER_ADDRESS_MODE_REPEAT, RG_SAMPLER_ADDRESS_MODE_REPEAT },
        true,
        debugName,
        std::nullopt,
        path );
}

uint32_t RTGL1::MetalTextureManager::AllocSlot()
{
    if( !freeSlots.empty() )
    {
        uint32_t s = freeSlots.back();
        freeSlots.pop_back();
        return s;
    }
    if( slots.size() >= MaxTextureCount )
    {
        return UINT32_MAX;
    }
    slots.emplace_back();
    return uint32_t( slots.size() - 1 );
}

void RTGL1::MetalTextureManager::FreeSlot( uint32_t slot )
{
    if( slot == EMPTY_TEXTURE_INDEX || slot >= slots.size() )
    {
        return;
    }
    // frames in flight might still sample it
    pendingDeletes.push_back( { slot, MAX_FRAMES_IN_FLIGHT + 1 } );
}

id< MTLSamplerState > RTGL1::MetalTextureManager::GetSampler( SamplerManager::Handle handle )
{
    auto [ addressU, addressV, filter ] = SamplerManager::Deconstruct( handle );
    // Deconstruct reports a dynamic filter as AUTO: it's the one from RgDrawFrameTexturesParams
    // (Vulkan's SamplerManager looks up the handle's index, after SetIfHasDynamicSamplerFilter)
    if( filter == RG_SAMPLER_FILTER_AUTO )
    {
        filter = dynamicFilter;
    }

    const bool     nearest = filter == RG_SAMPLER_FILTER_NEAREST;
    const uint32_t key     = ( nearest ? 1u : 0u ) | ( uint32_t( addressU ) << 1 ) | ( uint32_t( addressV ) << 2 );

    if( auto found = samplers.find( key ); found != samplers.end() )
    {
        return found->second;
    }

    // the same as SamplerManager.cpp
    MTLSamplerDescriptor* desc  = [MTLSamplerDescriptor new];
    desc.magFilter              = nearest ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    desc.minFilter              = forceMinificationFilterLinear || !nearest ? MTLSamplerMinMagFilterLinear
                                                                            : MTLSamplerMinMagFilterNearest;
    desc.mipFilter              = MTLSamplerMipFilterLinear;
    desc.sAddressMode           = ToMetal( addressU );
    desc.tAddressMode           = ToMetal( addressV );
    desc.maxAnisotropy          = 8;
    desc.supportArgumentBuffers = YES;
    if( @available( macOS 26.0, * ) )
    {
        desc.lodBias = mipLodBias;
    }

    id< MTLSamplerState > sampler = [device newSamplerStateWithDescriptor:desc];
    samplers[ key ]               = sampler;
    return sampler;
}

void RTGL1::MetalTextureManager::WriteTableEntry( uint32_t slot )
{
    const Slot& s = slots[ slot ];

    // an empty slot points to the white texture, so a stale index never reads freed memory
    const Slot& src = s.texture ? s : slots[ EMPTY_TEXTURE_INDEX ];

    const auto entry = TableEntry{
        .texture = src.texture.gpuResourceID,
        .sampler = GetSampler( src.sampler ).gpuResourceID,
    };
    static_cast< TableEntry* >( table.contents )[ slot ]           = entry;
    static_cast< MTLResourceID* >( imageTable.contents )[ slot ]   = entry.texture;
    static_cast< MTLResourceID* >( samplerTable.contents )[ slot ] = entry.sampler;
    liveTexturesDirty                                              = true;
}

uint32_t RTGL1::MetalTextureManager::PrepareTexture( const std::optional< ImageLoader::ResultInfo >& info,
                                                    SamplerManager::Handle                          sampler,
                                                    bool                                            useMipmaps,
                                                    const char*                                     debugName,
                                                    std::optional< RgTextureSwizzling >             swizzling,
                                                    const std::filesystem::path&                    filepath )
{
    if( !info )
    {
        return EMPTY_TEXTURE_INDEX;
    }
    if( info->baseSize.width == 0 || info->baseSize.height == 0 || !info->pData || info->levelCount == 0 )
    {
        debug::Warning( "Incorrect size ({},{}) of one of images in a texture {}",
                        info->baseSize.width,
                        info->baseSize.height,
                        Utils::SafeCstr( debugName ) );
        return EMPTY_TEXTURE_INDEX;
    }

    const MetalFormat format = ToMetalFormat( info->format );
    if( format.format == MTLPixelFormatInvalid )
    {
        debug::Warning( "Unsupported texture format {} of {}. Path: {}",
                        int( info->format ),
                        Utils::SafeCstr( debugName ),
                        filepath.string() );
        return EMPTY_TEXTURE_INDEX;
    }

    // SHIPPING_HACK begin: dont make mipmaps for voxel models
    if( filepath.string().contains( "vx_" ) )
    {
        useMipmaps = false;
    }
    // SHIPPING_HACK end

    const uint32_t width  = info->baseSize.width;
    const uint32_t height = info->baseSize.height;

    // TextureUploader: pregenerated levels are used as is, otherwise the mipmaps are generated
    uint32_t levelCount;
    bool     generateMipmaps = false;
    if( !useMipmaps )
    {
        levelCount = 1;
    }
    else if( info->isPregenerated && info->levelCount > 1 )
    {
        levelCount = std::min( info->levelCount, MAX_PREGENERATED_MIPMAP_LEVELS );
    }
    else if( format.IsCompressed() )
    {
        // can't render into a compressed texture
        levelCount = 1;
    }
    else
    {
        levelCount      = 1 + uint32_t( std::floor( std::log2( std::max( width, height ) ) ) );
        generateMipmaps = levelCount > 1;
    }

    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format.format
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:levelCount > 1];
    desc.mipmapLevelCount = levelCount;
    desc.usage            = MTLTextureUsageShaderRead;
    desc.storageMode      = MTLStorageModePrivate;
    desc.swizzle          = ToMetalSwizzle( swizzling );

    id< MTLTexture > texture = [device newTextureWithDescriptor:desc];
    if( !texture )
    {
        debug::Warning( "Failed to create a Metal texture for {}", Utils::SafeCstr( debugName ) );
        return EMPTY_TEXTURE_INDEX;
    }
    texture.label = [NSString stringWithUTF8String:Utils::SafeCstr( debugName )];

    const uint32_t slot = AllocSlot();
    if( slot == UINT32_MAX )
    {
        debug::Warning( "Reached texture limit: {}, while uploading {}", MaxTextureCount, Utils::SafeCstr( debugName ) );
        return EMPTY_TEXTURE_INDEX;
    }

    const uint32_t levelsToCopy = generateMipmaps ? 1 : levelCount;

    id< MTLBuffer > staging = [device newBufferWithBytes:info->pData
                                                  length:info->dataSize
                                                 options:MTLResourceStorageModeShared];

    // the same queue as the frames, so the upload finishes before a frame that samples it
    id< MTLCommandBuffer >      cmd  = [queue commandBuffer];
    id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
    for( uint32_t level = 0; level < levelsToCopy; level++ )
    {
        const uint32_t   w           = std::max( 1u, width >> level );
        const uint32_t   h           = std::max( 1u, height >> level );
        const uint32_t   blocksWide  = ( w + format.blockSize - 1 ) / format.blockSize;
        const uint32_t   blocksHigh  = ( h + format.blockSize - 1 ) / format.blockSize;
        const NSUInteger bytesPerRow = NSUInteger( blocksWide ) * format.bytesPerBlock;

        const size_t offset = info->levelOffsets[ level ];
        if( offset + bytesPerRow * blocksHigh > info->dataSize )
        {
            debug::Warning( "Texture level {} of {} is out of the data bounds", level, Utils::SafeCstr( debugName ) );
            break;
        }

        [blit copyFromBuffer:staging
                   sourceOffset:offset
              sourceBytesPerRow:bytesPerRow
            sourceBytesPerImage:bytesPerRow * blocksHigh
                     sourceSize:MTLSizeMake( w, h, 1 )
                      toTexture:texture
               destinationSlice:0
               destinationLevel:level
              destinationOrigin:MTLOriginMake( 0, 0, 0 )];
    }
    if( generateMipmaps )
    {
        [blit generateMipmapsForTexture:texture];
    }
    [blit endEncoding];
    [cmd commit];

    slots[ slot ] = Slot{
        .texture = texture,
        .sampler = sampler,
    };
    WriteTableEntry( slot );

    if( info->isPregenerated )
    {
        debug::Verbose( "Texture file: {}", filepath.string() );
    }
    return slot;
}

void RTGL1::MetalTextureManager::MakeMaterial( std::string_view                                 materialName,
                                               std::span< TextureOverrides >                    ovrd,
                                               std::span< const SamplerManager::Handle >        samplerHandles,
                                               std::span< std::optional< RgTextureSwizzling > > swizzlings )
{
    assert( ovrd.size() == TEXTURES_PER_MATERIAL_COUNT );
    assert( samplerHandles.size() == TEXTURES_PER_MATERIAL_COUNT );
    assert( swizzlings.size() == TEXTURES_PER_MATERIAL_COUNT );

    MaterialTextures mtextures = {};
    for( uint32_t i = 0; i < TEXTURES_PER_MATERIAL_COUNT; i++ )
    {
        mtextures.indices[ i ] = PrepareTexture(
            ovrd[ i ].result, samplerHandles[ i ], true, ovrd[ i ].debugname, swizzlings[ i ], ovrd[ i ].path );
    }

    // TextureManager::InsertMaterial: replace an existing one
    TryDestroyMaterial( std::string( materialName ).c_str() );
    materials.emplace( std::string( materialName ), Material{ .textures = mtextures } );
}

void RTGL1::MetalTextureManager::Provide( const RgOriginalTextureInfo& info )
{
    if( info.sType != RG_STRUCTURE_TYPE_ORIGINAL_TEXTURE_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    if( Utils::IsCstrEmpty( info.pTextureName ) )
    {
        debug::Warning( "RgOriginalTextureInfo::pTextureName must not be null or an empty string" );
        return;
    }
    if( info.pPixels == nullptr )
    {
        debug::Warning( "RgOriginalTextureInfo::pPixels must not be null" );
        return;
    }

    // promote material from 'imported' to 'original',
    // if a game creates a material, so it's not deleted in FreeAllImportedMaterials
    if( importedMaterials.erase( info.pTextureName ) )
    {
        debug::Verbose( R"(Material is promoted from 'Imported' to 'Original': {})", info.pTextureName );
    }

    if( PreferExistingMaterials )
    {
        if( materials.contains( info.pTextureName ) )
        {
            debug::Verbose( "Material with the same name already exists, ignoring new data: {}", info.pTextureName );
            return;
        }
    }

    const auto* details = pnext::find< RgOriginalTextureDetailsEXT >( &info );

    // clang-format off
    TextureOverrides ovrd[] = {
        TextureOverrides{ ovrdFolder, info.pTextureName, Postfixes[ 0 ], info.pPixels, info.size, GetFormat( details, VK_FORMAT_R8G8B8A8_SRGB ), OnlyKTX2LoaderIfNonDevMode() },
        TextureOverrides{ ovrdFolder, info.pTextureName, Postfixes[ 1 ], nullptr, {}, VK_FORMAT_R8G8B8A8_UNORM, OnlyKTX2LoaderIfNonDevMode() },
        TextureOverrides{ ovrdFolder, info.pTextureName, Postfixes[ 2 ], nullptr, {}, VK_FORMAT_R8G8B8A8_UNORM, OnlyKTX2LoaderIfNonDevMode() },
        TextureOverrides{ ovrdFolder, info.pTextureName, Postfixes[ 3 ], nullptr, {}, VK_FORMAT_R8G8B8A8_SRGB, OnlyKTX2LoaderIfNonDevMode() },
        TextureOverrides{ ovrdFolder, info.pTextureName, Postfixes[ 4 ], nullptr, {}, VK_FORMAT_R8_UNORM, OnlyKTX2LoaderIfNonDevMode() },
    };
    static_assert( std::size( ovrd ) == TEXTURES_PER_MATERIAL_COUNT );
    // clang-format on

    const SamplerManager::Handle samplerHandles[] = {
        SamplerManager::Handle{ info.filter, info.addressModeU, info.addressModeV },
        SamplerManager::Handle{ info.filter, info.addressModeU, info.addressModeV },
        SamplerManager::Handle{
            forceNormalMapFilterLinear ? RG_SAMPLER_FILTER_LINEAR : info.filter, info.addressModeU, info.addressModeV },
        SamplerManager::Handle{ info.filter, info.addressModeU, info.addressModeV },
        SamplerManager::Handle{ RG_SAMPLER_FILTER_LINEAR, info.addressModeU, info.addressModeV },
    };
    static_assert( std::size( samplerHandles ) == TEXTURES_PER_MATERIAL_COUNT );
    static_assert( TEXTURE_NORMAL_INDEX == 2 );

    std::optional< RgTextureSwizzling > swizzlings[] = {
        std::nullopt, std::optional{ pbrSwizzling }, std::nullopt, std::nullopt, std::nullopt,
    };
    static_assert( TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_INDEX == 1 );

    MakeMaterial( info.pTextureName, ovrd, samplerHandles, swizzlings );
}

bool RTGL1::MetalTextureManager::TryCreateImportedMaterial( const std::string&                        materialName,
                                                            std::span< const std::filesystem::path >  fullPaths,
                                                            std::span< const SamplerManager::Handle > samplerHandles,
                                                            RgTextureSwizzling                        customPbrSwizzling,
                                                            bool                                      isReplacement )
{
    assert( fullPaths.size() == TEXTURES_PER_MATERIAL_COUNT );
    assert( samplerHandles.size() == TEXTURES_PER_MATERIAL_COUNT );

    if( materialName.empty() )
    {
        return false;
    }

    // check if already uploaded
    if( auto found = importedMaterials.find( materialName ); found != importedMaterials.end() )
    {
        if( isReplacement )
        {
            // promote to a stronger type
            found->second = ImportedType::ForReplacement;
        }
        return true;
    }

    if( PreferExistingMaterials )
    {
        if( materials.contains( materialName ) )
        {
            debug::Verbose( "Material with the same name already exists, ignoring new data: {}", materialName );
            return false;
        }
    }

    if( std::ranges::all_of( fullPaths, []( auto&& p ) { return p.empty(); } ) )
    {
        return false;
    }
    if( std::ranges::none_of( fullPaths, []( auto&& p ) { return std::filesystem::is_regular_file( p ); } ) )
    {
        debug::Warning( "Fail to create imported material: none of the paths lead to a file:"
                        "\n  A: {}\n  B: {}\n  C: {}\n  D: {}\n  E: {}\n",
                        fullPaths[ 0 ].string(),
                        fullPaths[ 1 ].string(),
                        fullPaths[ 2 ].string(),
                        fullPaths[ 3 ].string(),
                        fullPaths[ 4 ].string() );
        return false;
    }

    // clang-format off
    TextureOverrides ovrd[] = {
        TextureOverrides{ fullPaths[ 0 ], true, AnyImageLoader() },
        TextureOverrides{ fullPaths[ 1 ], false, AnyImageLoader() },
        TextureOverrides{ fullPaths[ 2 ], false, AnyImageLoader() },
        TextureOverrides{ fullPaths[ 3 ], true, AnyImageLoader() },
        TextureOverrides{ fullPaths[ 4 ], true, AnyImageLoader() },
    };
    static_assert( std::size( ovrd ) == TEXTURES_PER_MATERIAL_COUNT );
    // clang-format on

    std::optional< RgTextureSwizzling > swizzlings[] = {
        std::nullopt, std::optional{ customPbrSwizzling }, std::nullopt, std::nullopt, std::nullopt,
    };

    importedMaterials.emplace( materialName, isReplacement ? ImportedType::ForReplacement : ImportedType::ForStatic );

    MakeMaterial( materialName, ovrd, samplerHandles, swizzlings );
    return true;
}

void RTGL1::MetalTextureManager::FreeAllImportedMaterials( bool freeReplacements )
{
    if( freeReplacements )
    {
        for( const auto& materialName : importedMaterials | std::views::keys )
        {
            TryDestroyMaterial( materialName.c_str() );
        }
        importedMaterials.clear();
    }
    else
    {
        for( const auto& [ materialName, importType ] : importedMaterials )
        {
            if( importType != ImportedType::ForReplacement )
            {
                TryDestroyMaterial( materialName.c_str() );
            }
        }
        std::erase_if( importedMaterials, []( const auto& kv ) { return kv.second != ImportedType::ForReplacement; } );
    }
}

bool RTGL1::MetalTextureManager::TryDestroyMaterial( const char* materialName )
{
    if( Utils::IsCstrEmpty( materialName ) )
    {
        return false;
    }
    auto found = materials.find( materialName );
    if( found == materials.end() )
    {
        return false;
    }
    for( uint32_t index : found->second.textures.indices )
    {
        FreeSlot( index );
    }
    materials.erase( found );
    return true;
}

void RTGL1::MetalTextureManager::MarkAsDeleted( const char* pTextureName )
{
    TryDestroyMaterial( pTextureName );
}

auto RTGL1::MetalTextureManager::GetMaterialTextures( const char* materialName ) const -> MaterialTextures
{
    if( Utils::IsCstrEmpty( materialName ) )
    {
        return EmptyMaterialTextures;
    }
    auto found = materials.find( materialName );
    return found != materials.end() ? found->second.textures : EmptyMaterialTextures;
}

auto RTGL1::MetalTextureManager::GetTexturesForLayers( const RgMeshPrimitiveInfo& primitive ) const
    -> std::array< MaterialTextures, 4 >
{
    auto layers = pnext::find< RgMeshPrimitiveTextureLayersEXT >( &primitive );

    return {
        GetMaterialTextures( primitive.pTextureName ),
        GetMaterialTextures( layers && layers->pLayer1 ? layers->pLayer1->pTextureName : nullptr ),
        GetMaterialTextures( layers && layers->pLayer2 ? layers->pLayer2->pTextureName : nullptr ),
        GetMaterialTextures( layers && layers->pLayer3 ? layers->pLayer3->pTextureName : nullptr ),
    };
}

auto RTGL1::MetalTextureManager::GetColorForLayers( const RgMeshPrimitiveInfo& primitive ) const
    -> std::array< RgColor4DPacked32, 4 >
{
    auto layers = pnext::find< RgMeshPrimitiveTextureLayersEXT >( &primitive );

    return {
        primitive.color,
        layers && layers->pLayer1 ? layers->pLayer1->color : 0xFFFFFFFF,
        layers && layers->pLayer2 ? layers->pLayer2->color : 0xFFFFFFFF,
        layers && layers->pLayer3 ? layers->pLayer3->color : 0xFFFFFFFF,
    };
}

uint32_t RTGL1::MetalTextureManager::Find( const char* pTextureName ) const
{
    return GetMaterialTextures( pTextureName ).indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
}

void RTGL1::MetalTextureManager::OnFrameStart()
{
    for( auto& p : pendingDeletes )
    {
        if( --p.framesLeft == 0 )
        {
            slots[ p.slot ] = Slot{};
            WriteTableEntry( p.slot );
            freeSlots.push_back( p.slot );
        }
    }
    std::erase_if( pendingDeletes, []( const PendingDelete& p ) { return p.framesLeft == 0; } );
}

void RTGL1::MetalTextureManager::SetDynamicSamplerFilter( RgSamplerFilter filter )
{
    if( filter == RG_SAMPLER_FILTER_AUTO )
    {
        filter = RG_SAMPLER_FILTER_LINEAR;
    }
    if( filter == dynamicFilter )
    {
        return;
    }
    dynamicFilter = filter;

    for( uint32_t i = 0; i < slots.size(); i++ )
    {
        if( slots[ i ].texture )
        {
            WriteTableEntry( i );
        }
    }
}

void RTGL1::MetalTextureManager::SetMipLodBias( float bias )
{
    if( std::abs( bias - mipLodBias ) < 0.01f )
    {
        return;
    }
    mipLodBias = bias;

    // the frames in flight keep the old samplers alive, as they're referenced by the command buffers
    samplers.clear();
    for( uint32_t i = 0; i < slots.size(); i++ )
    {
        if( slots[ i ].texture )
        {
            WriteTableEntry( i );
        }
    }
}

const std::vector< id< MTLResource > >& RTGL1::MetalTextureManager::LiveTextures() const
{
    if( liveTexturesDirty )
    {
        liveTextures.clear();
        for( const Slot& s : slots )
        {
            if( s.texture )
            {
                liveTextures.push_back( s.texture );
            }
        }
        liveTexturesDirty = false;
    }
    return liveTextures;
}

void RTGL1::MetalTextureManager::UseResources( id< MTLComputeCommandEncoder > enc ) const
{
    // TODO: MTLResidencySet, to not list every texture each frame
    const auto& live = LiveTextures();
    [enc useResources:live.data() count:live.size() usage:MTLResourceUsageRead];
}

void RTGL1::MetalTextureManager::UseResources( id< MTLRenderCommandEncoder > enc ) const
{
    const auto& live = LiveTextures();
    [enc useResources:live.data() count:live.size() usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
}
