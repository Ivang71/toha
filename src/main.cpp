#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "core/logging.hpp"
#include "render/vulkan/vulkan_app.hpp"

int main(int argc, char** argv) {
#ifndef NDEBUG
    bool enableDebug = true;
#else
    bool enableDebug = false;
#endif

    gLogFile.open("voxel_engine.log", std::ios::out | std::ios::trunc);
    if (gLogFile.is_open()) {
        gLogFile << "voxel_engine start\n";
        gLogFile.flush();
    }

    if (std::getenv("VOXEL_VK_DEBUG")) {
        enableDebug = true;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vk-debug") enableDebug = true;
        if (arg == "--vk-nodebug") enableDebug = false;
    }

        VulkanApp app(enableDebug);
        app.run();
    return 0;
}

