#include "render/vulkan/app/vulkan_app_impl.hpp"

#include <cstring>
#include <stdexcept>
#include <cmath>

void VulkanAppImpl::createCameraBuffer() {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(CameraUBO);
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &info, nullptr, &cameraBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create camera buffer");
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, cameraBuffer, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &alloc, nullptr, &cameraBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate camera buffer memory");
    }

    vkBindBufferMemory(device, cameraBuffer, cameraBufferMemory, 0);
}

void VulkanAppImpl::initCamera() {
    cameraPos = {0.0f, 1.5f, 6.0f};
    lastLoggedCameraPos = cameraPos;
    cameraYaw = -1.5707963f;
    cameraPitch = 0.0f;
    firstMouse = true;
    cameraForward = vnorm({std::cos(cameraPitch) * std::cos(cameraYaw),
                           std::sin(cameraPitch),
                           std::cos(cameraPitch) * std::sin(cameraYaw)});
    Vec3 up = {0.0f, 1.0f, 0.0f};
    cameraRight = vnorm(vcross(cameraForward, up));
    cameraUp = vcross(cameraRight, cameraForward);

    float fov = 1.0471976f;
    float aspect = static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height);
    float sliceY = 0.0f;

    cameraData.camPos[0] = cameraPos.x;
    cameraData.camPos[1] = cameraPos.y;
    cameraData.camPos[2] = cameraPos.z;
    cameraData.camPos[3] = 1.0f;

    cameraData.camForward[0] = cameraForward.x;
    cameraData.camForward[1] = cameraForward.y;
    cameraData.camForward[2] = cameraForward.z;
    cameraData.camForward[3] = 0.0f;

    cameraData.camRight[0] = cameraRight.x;
    cameraData.camRight[1] = cameraRight.y;
    cameraData.camRight[2] = cameraRight.z;
    cameraData.camRight[3] = 0.0f;

    cameraData.camUp[0] = cameraUp.x;
    cameraData.camUp[1] = cameraUp.y;
    cameraData.camUp[2] = cameraUp.z;
    cameraData.camUp[3] = 0.0f;

    cameraData.params[0] = fov;
    cameraData.params[1] = aspect;
    cameraData.params[2] = sliceY;
    cameraData.params[3] = 0.0f;

    updateCameraBuffer();
}

void VulkanAppImpl::updateCamera(float dt) {
    if (dt <= 0.0f) dt = 0.016f;

    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (firstMouse) {
        lastMouseX = x;
        lastMouseY = y;
        firstMouse = false;
    }
    double dx = x - lastMouseX;
    double dy = y - lastMouseY;
    lastMouseX = x;
    lastMouseY = y;

    float sensitivity = 0.002f;
    cameraYaw -= static_cast<float>(dx) * sensitivity;
    cameraPitch -= static_cast<float>(dy) * sensitivity;
    const float limit = 1.55334f;
    if (cameraPitch > limit) cameraPitch = limit;
    if (cameraPitch < -limit) cameraPitch = -limit;

    cameraForward = vnorm({std::cos(cameraPitch) * std::cos(cameraYaw),
                           std::sin(cameraPitch),
                           std::cos(cameraPitch) * std::sin(cameraYaw)});
    Vec3 up = {0.0f, 1.0f, 0.0f};
    cameraRight = vnorm(vcross(cameraForward, up));
    cameraUp = vcross(cameraRight, cameraForward);

    float speed = 10.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        speed *= 3.0f;
    }
    float vel = speed * dt;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos = vadd(cameraPos, vscale(cameraForward, vel));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPos = vsub(cameraPos, vscale(cameraForward, vel));
    }
    Vec3 flatRight = {cameraRight.x, 0.0f, cameraRight.z};
    flatRight = vnorm(flatRight);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPos = vsub(cameraPos, vscale(flatRight, vel));
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPos = vadd(cameraPos, vscale(flatRight, vel));
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        cameraPos.y += vel;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        cameraPos.y -= vel;
    }

    Vec3 delta = vsub(cameraPos, lastLoggedCameraPos);
    if (vlen(delta) > 0.001f) {
        lastLoggedCameraPos = cameraPos;
        std::lock_guard<std::mutex> lock(gLogMutex);
    }
}

void VulkanAppImpl::updateCameraBuffer() {
    cameraData.camPos[0] = cameraPos.x;
    cameraData.camPos[1] = cameraPos.y;
    cameraData.camPos[2] = cameraPos.z;

    cameraData.camForward[0] = cameraForward.x;
    cameraData.camForward[1] = cameraForward.y;
    cameraData.camForward[2] = cameraForward.z;

    cameraData.camRight[0] = cameraRight.x;
    cameraData.camRight[1] = cameraRight.y;
    cameraData.camRight[2] = cameraRight.z;

    cameraData.camUp[0] = cameraUp.x;
    cameraData.camUp[1] = cameraUp.y;
    cameraData.camUp[2] = cameraUp.z;

    float aspect = static_cast<float>(swapchainExtent.width) / static_cast<float>(swapchainExtent.height);
    cameraData.params[1] = aspect;

    void* data = nullptr;
    vkMapMemory(device, cameraBufferMemory, 0, sizeof(CameraUBO), 0, &data);
    std::memcpy(data, &cameraData, sizeof(CameraUBO));
    vkUnmapMemory(device, cameraBufferMemory);
}

