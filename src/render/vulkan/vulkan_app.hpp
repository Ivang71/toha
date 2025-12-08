#pragma once

class VulkanAppImpl;

class VulkanApp {
public:
    explicit VulkanApp(bool enableValidation);
    ~VulkanApp();

    void run();

private:
    VulkanAppImpl* impl;
};


