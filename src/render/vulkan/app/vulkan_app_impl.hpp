#pragma once

#include "render/vulkan/vulkan_debug.hpp"
#include "core/logging.hpp"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <optional>
#include <cstdint>
#include <mutex>
#include <cmath>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vec3 {
    float x;
    float y;
    float z;
};

struct CameraUBO {
    float camPos[4];
    float camForward[4];
    float camRight[4];
    float camUp[4];
    float params[4];
};

inline Vec3 vadd(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 vsub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 vscale(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 vcross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float vlen(Vec3 a) { return std::sqrt(vdot(a, a)); }
inline Vec3 vnorm(Vec3 a) {
    float len = vlen(a);
    if (len <= 0.0f) return {0.0f, 0.0f, 0.0f};
    float inv = 1.0f / len;
    return {a.x * inv, a.y * inv, a.z * inv};
}

class VulkanAppImpl {
public:
    explicit VulkanAppImpl(bool enableValidation);
    ~VulkanAppImpl() = default;

    void run();

    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice dev);
    uint64_t rateDevice(VkPhysicalDevice dev);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev);
    bool checkDeviceExtensionSupport(VkPhysicalDevice dev);
    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice dev);
    void createLogicalDevice();
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void createSwapchain();
    void createImageViews();
    static std::vector<char> readFile(const char* filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createComputeDescriptorSetLayout();
    void createComputePipeline();
    void createComputeDescriptorPool();
    void createComputeDescriptorSets();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void drawFrame();
    void createCameraBuffer();
    void initCamera();
    void updateCamera(float dt);
    void updateCameraBuffer();

private:
    GLFWwindow* window{};
    VkInstance instance{};
    VkDebugUtilsMessengerEXT debugMessenger{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physicalDevice{};
    VkDevice device{};
    VkQueue graphicsQueue{};
    VkQueue presentQueue{};
    VkSwapchainKHR swapchain{};
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainImageFormat{};
    VkExtent2D swapchainExtent{};
    std::vector<VkImageView> swapchainImageViews;
    VkPipelineLayout computePipelineLayout{};
    VkPipeline computePipeline{};
    VkDescriptorSetLayout computeDescriptorSetLayout{};
    VkDescriptorPool computeDescriptorPool{};
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkCommandPool commandPool{};
    std::vector<VkCommandBuffer> commandBuffers;
    VkSemaphore imageAvailableSemaphore{};
    VkSemaphore renderFinishedSemaphore{};
    VkFence inFlightFence{};

    VkBuffer cameraBuffer{};
    VkDeviceMemory cameraBufferMemory{};
    CameraUBO cameraData{};
    Vec3 cameraPos{};
    Vec3 lastLoggedCameraPos{};
    Vec3 cameraForward{};
    Vec3 cameraRight{};
    Vec3 cameraUp{};
    float cameraYaw{};
    float cameraPitch{};
    bool firstMouse{};
    double lastMouseX{};
    double lastMouseY{};

    std::vector<bool> imageLayoutInitialized;

    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;
    const uint32_t RAYMARCH_UPSCALE = 2;
    bool validationEnabled{};
    bool cursorLocked{};
    double fpsTimeAccum{};
    int fpsFrameCount{};
};

