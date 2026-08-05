#include <iostream>

// 1. Define implementation macros in EXACTLY ONE .cpp file before headers
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// 2. Include the metal-cpp headers
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

int main() {
    // 3. Create a top-level Autorelease Pool for memory management
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // 4. Initialize the default GPU device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Error: Metal is not supported on this system.\n";
        pool->release();
        return -1;
    }

    // 5. Print out the GPU name (converts native Apple string to C-style string)
    std::cout << "-------------------------------------------\n";
    std::cout << " Metal successfully initialized via C++!   \n";
    std::cout << " GPU Device Name: " << device->name()->utf8String() << "\n";
    std::cout << "-------------------------------------------\n";

    // 6. Clean up allocated object memory in reverse order
    device->release();
    pool->release();

    return 0;
}

