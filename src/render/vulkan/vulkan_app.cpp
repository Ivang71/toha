// Thin wrapper that delegates to the implementation class.
#include "render/vulkan/vulkan_app.hpp"
#include "render/vulkan/app/vulkan_app_impl.hpp"

VulkanApp::VulkanApp(bool enableValidation)
    : impl(new VulkanAppImpl(enableValidation)) {}

VulkanApp::~VulkanApp() { delete impl; }

void VulkanApp::run() { impl->run(); }


