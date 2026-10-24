#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

// Makes the GLFW window's content view layer-backed by a CAMetalLayer, for RgMetalSurfaceCreateInfo.
// RTGL1 sets the layer's device and pixel format, and follows its size.
void* RtglTest_AttachMetalLayer( void* nsWindow )
{
    NSWindow* window = ( __bridge NSWindow* )nsWindow;
    NSView*   view   = window.contentView;

    CAMetalLayer* layer   = [CAMetalLayer layer];
    layer.contentsScale   = window.backingScaleFactor;
    layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;

    view.layer      = layer;
    view.wantsLayer = YES;

    // the view keeps the layer alive
    return ( __bridge void* )layer;
}
