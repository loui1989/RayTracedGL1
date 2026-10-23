// Copyright (c) 2026 Quake RT for macOS contributors
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

// Builds without DX12 (RG_WITH_DX12=OFF, e.g. macOS): the DXGI swapchain, FSR3 and DLSS3
// frame generation report themselves unavailable. Replaces DX12_Vulkan.cpp,
// DX12_Swapchain.cpp, FSR3_DX12.cpp and DLSS3_DX12.cpp.

#ifndef RG_USE_DX12

#include "DLSS3_DX12.h"
#include "DX12_Interop.h"
#include "FSR3_DX12.h"

#include <cassert>

namespace RTGL1::dxgi
{

bool DX12Supported() { return false; }
bool HasDX12Instance() { return false; }
bool HasDX12SwapchainInstance() { return false; }

void SetHwnd( void* ) {}
void SetVk( VkDevice, VkPhysicalDevice ) {}

bool InitAsFSR3( uint64_t, PFN_CreateSwapchain ) { return false; }
bool InitAsDLFG( uint64_t, PFN_SetD3D12, PFN_UpgradeInterface, PFN_GetNativeInterface ) { return false; }
auto InitAsRawDXGI( uint64_t ) -> std::expected< bool, const char* >
{
    return std::unexpected( "DX12 is not available in this build" );
}
bool HasRawDXGI() { return false; }
void Destroy() {}
auto GetD3D12Device() -> ID3D12Device* { return nullptr; }
auto GetD3D12CommandQueue() -> ID3D12CommandQueue* { return nullptr; }
auto GetAdapterLUID() -> uint64_t { return 0; }
void WaitAndPrepareForFrame( ID3D12Fence*, HANDLE, uint64_t ) {}
auto CreateD3D12CommandList( uint32_t ) -> ID3D12GraphicsCommandList* { return nullptr; }

void DispatchBlit( ID3D12GraphicsCommandList*, ID3D12Resource*, ID3D12Resource*, uint32_t, uint32_t, bool ) {}

auto CreateSwapchain( uint32_t, uint32_t, uint32_t, int, int, bool ) -> uint32_t { return 0; }
void DestroySwapchain( bool ) {}
auto GetSwapchainBack( uint32_t ) -> ID3D12Resource* { return nullptr; }
auto GetSwapchainCopySrc( uint32_t* width, uint32_t* height, bool* convertToSrgb ) -> ID3D12Resource*
{
    if( width )
    {
        *width = 0;
    }
    if( height )
    {
        *height = 0;
    }
    if( convertToSrgb )
    {
        *convertToSrgb = false;
    }
    return nullptr;
}
auto GetSwapchainDxgiFormat() -> int { return 0; }
auto GetSwapchainDxgiSwapchain() -> IDXGISwapChain4* { return nullptr; }
void Present( ID3D12Fence*, uint64_t ) {}
auto GetCurrentBackBufferIndex() -> uint32_t
{
    assert( 0 );
    return 0;
}
void WaitIdle() {}

auto Semaphores_GetVkDx12Shared( SharedSemaphoreType ) -> std::optional< SharedSemaphore >
{
    return std::nullopt;
}

void Framebuf_CreateDX12Resources( CommandBufferManager&, MemoryAllocator&, const ResolutionState& ) {}
void Framebuf_Destroy() {}
auto Framebuf_GetVkDx12Shared( int ) -> SharedImage { return {}; }
bool Framebuf_HasSharedImages() { return false; }

}

namespace RTGL1
{

// ---- FSR3 frame generation

void FSR3_DX12::LoadSDK() {}
void FSR3_DX12::UnloadSDK() {}

auto FSR3_DX12::MakeInstance( uint64_t ) -> std::expected< FSR3_DX12*, const char* >
{
    return std::unexpected( "FSR3 frame generation needs DX12, not available in this build" );
}

FSR3_DX12::~FSR3_DX12() = default;

void FSR3_DX12::CopyVkInputsToDX12( VkCommandBuffer, uint32_t, const Framebuffers&, const ResolutionState& ) {}

auto FSR3_DX12::Apply( ID3D12CommandList*,
                       uint32_t,
                       const Framebuffers&,
                       const RenderResolutionHelper&,
                       RgFloat2D,
                       double,
                       float,
                       float,
                       float,
                       bool,
                       float,
                       bool ) -> std::optional< FramebufferImageIndex >
{
    return std::nullopt;
}

void FSR3_DX12::CopyDX12OutputToVk( VkCommandBuffer, uint32_t, const Framebuffers&, const ResolutionState& ) {}

RgFloat2D FSR3_DX12::GetJitter( const ResolutionState&, uint32_t ) const
{
    return { 0, 0 };
}

void FSR3_DX12::OnFramebuffersSizeChange( const ResolutionState& ) {}

// ---- DLSS3 frame generation

void DLSS3_DX12::LoadSDK( const char* ) {}
void DLSS3_DX12::UnloadSDK() {}

auto DLSS3_DX12::MakeInstance( uint64_t, bool ) -> std::expected< DLSS3_DX12*, const char* >
{
    return std::unexpected( "DLSS3 frame generation needs DX12, not available in this build" );
}

DLSS3_DX12::~DLSS3_DX12() = default;

void DLSS3_DX12::CopyVkInputsToDX12( VkCommandBuffer, uint32_t, const Framebuffers&, const ResolutionState& ) {}

auto DLSS3_DX12::Apply( ID3D12CommandList*,
                        uint32_t,
                        Framebuffers&,
                        const RenderResolutionHelper&,
                        RgFloat2D,
                        double,
                        bool,
                        const Camera&,
                        uint32_t,
                        bool ) -> std::optional< FramebufferImageIndex >
{
    return std::nullopt;
}

void DLSS3_DX12::CopyDX12OutputToVk( VkCommandBuffer, uint32_t, const Framebuffers&, const ResolutionState& ) {}

auto DLSS3_DX12::GetOptimalSettings( uint32_t userWidth, uint32_t userHeight, RgRenderResolutionMode ) const
    -> std::pair< uint32_t, uint32_t >
{
    return { userWidth, userHeight };
}

void DLSS3_DX12::Reflex_SimStart( uint32_t ) {}
void DLSS3_DX12::Reflex_SimEnd() {}
void DLSS3_DX12::Reflex_RenderStart() {}
void DLSS3_DX12::Reflex_RenderEnd() {}
void DLSS3_DX12::Reflex_PresentStart() {}
void DLSS3_DX12::Reflex_PresentEnd() {}

}

#endif // !RG_USE_DX12
