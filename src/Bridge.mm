#import "Bridge.hpp"
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Metal/Metal.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

void setupMetalLayerForWindow(GLFWwindow* window, void* metalDevicePtr) {
    NSWindow* nativeWin = (NSWindow*)glfwGetCocoaWindow(window);
    
    // Create native CAMetalLayer using Objective-C
    CAMetalLayer* metalLayer = [CAMetalLayer layer];
    
    // Safely cast the void* back to an Objective-C Metal device object
    id<MTLDevice> device = (__bridge id<MTLDevice>)metalDevicePtr;
    
    [metalLayer setDevice:device];
    [metalLayer setPixelFormat:MTLPixelFormatBGRA8Unorm];
    
    // Attach the layer to the window
    [nativeWin.contentView setLayer:metalLayer];
    [nativeWin.contentView setWantsLayer:YES];
}
