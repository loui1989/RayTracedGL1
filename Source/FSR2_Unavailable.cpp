// AMD FSR2 off Windows: its runtime here is Windows DLLs (Source/bin), so on
// Linux the upscaler slot is empty (MakeInstance gives none, as when the DLLs
// are missing on Windows).

#include "FSR2.h"

RTGL1::FSR2::FSR2( VkDevice _device, VkPhysicalDevice _physDevice )
    : device{ _device }, physDevice{ _physDevice }
{
}

RTGL1::FSR2::~FSR2() = default;

bool RTGL1::FSR2::Valid() const
{
    return false;
}

auto RTGL1::FSR2::MakeInstance( VkDevice, VkPhysicalDevice ) -> std::shared_ptr< FSR2 >
{
    return {};
}

void RTGL1::FSR2::OnFramebuffersSizeChange( const ResolutionState& ) {}

RTGL1::FramebufferImageIndex RTGL1::FSR2::Apply( VkCommandBuffer,
                                                 uint32_t,
                                                 const Framebuffers&,
                                                 const RenderResolutionHelper&,
                                                 RgFloat2D,
                                                 double,
                                                 float,
                                                 float,
                                                 float,
                                                 bool,
                                                 float )
{
    return FramebufferImageIndex::FB_IMAGE_INDEX_FINAL;
}

RgFloat2D RTGL1::FSR2::GetJitter( const ResolutionState&, uint32_t ) const
{
    return {};
}
