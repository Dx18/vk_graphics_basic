#ifndef SIMPLE_INSTANCES_RENDER_H
#define SIMPLE_INSTANCES_RENDER_H

#include "../../render/scene_mgr.h"
#include "../../render/render_common.h"
#include "../../../resources/shaders/common.h"
#include "etna/GraphicsPipeline.hpp"
#include <geom/vk_mesh.h>
#include <vk_descriptor_sets.h>
#include <vk_fbuf_attachment.h>
#include <vk_images.h>
#include <vk_swapchain.h>
#include <vk_quad.h>

#include <string>
#include <iostream>

#include <etna/GlobalContext.hpp>
#include <etna/Sampler.hpp>


class IRenderGUI;

class InstancesRender : public IRender
{
public:
  InstancesRender(uint32_t a_width, uint32_t a_height, uint32_t a_instanceCount);
  ~InstancesRender();

  uint32_t GetWidth() const override { return m_width; }
  uint32_t GetHeight() const override { return m_height; }
  VkInstance GetVkInstance() const override { return m_context->getInstance(); }

  void InitVulkan(const char **a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId) override;

  void InitPresentation(VkSurfaceKHR &a_surface, bool initGUI) override;

  void ProcessInput(const AppInput &input) override;
  void UpdateCamera(const Camera *cams, uint32_t a_camsNumber) override;
  Camera GetCurrentCamera() override { return m_cam; }
  void UpdateView();

  void LoadScene(const char *path, bool transpose_inst_matrices) override;
  void DrawFrame(float a_time, DrawMode a_mode) override;

private:
  etna::GlobalContext *m_context = nullptr;
  etna::Image m_mainViewDepth;
  etna::Sampler m_defaultSampler;

  etna::Buffer m_matrixBuffer;
  etna::Buffer m_indexBuffer;
  etna::Buffer m_drawCommandBuffer;

  VkCommandPool m_commandPool = VK_NULL_HANDLE;

  struct
  {
    uint32_t currentFrame         = 0u;
    VkQueue queue                 = VK_NULL_HANDLE;
    VkSemaphore imageAvailable    = VK_NULL_HANDLE;
    VkSemaphore renderingFinished = VK_NULL_HANDLE;
  } m_presentationResources;

  std::vector<VkFence> m_frameFences;
  std::vector<VkCommandBuffer> m_cmdBuffersDrawMain;

  float4x4 m_worldViewProj;

  etna::ComputePipeline m_frustumCullingPipeline{};
  etna::GraphicsPipeline m_simpleMaterialPipeline{};

  std::shared_ptr<vk_utils::DescriptorMaker> m_pFrustumCullingBindings = nullptr;
  std::shared_ptr<vk_utils::DescriptorMaker> m_pSimpleMaterialBindings = nullptr;

  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VulkanSwapChain m_swapchain;

  Camera m_cam;
  uint32_t m_width = 1024u;
  uint32_t m_height = 1024u;
  uint32_t m_framesInFlight = 2u;
  bool m_vsync = false;

  uint32_t m_instanceCount = 10000;

  vk::PhysicalDeviceFeatures m_enabledDeviceFeatures = {};
  std::vector<const char *> m_deviceExtensions;
  std::vector<const char *> m_instanceExtensions;

  std::shared_ptr<SceneManager> m_pScnMgr;
  std::shared_ptr<IRenderGUI> m_pGUIRender;

  VkDescriptorSet m_frustumCullingDS = nullptr;
  VkDescriptorSetLayout m_frustumCullingDSLayout = nullptr;
  VkDescriptorSet m_simpleMaterialDS = nullptr;
  VkDescriptorSetLayout m_simpleMaterialDSLayout = nullptr;

  struct
  {
    float4x4 mViewProjection;
    float4 boundingBoxBegin;
    float4 boundingBoxEnd;
    uint32_t length;
  } m_frustumCullingParams;

  struct
  {
    float4x4 mViewProjection;
  } m_renderParams;

  void DrawFrameSimple(bool draw_gui);

  //void CreateInstance();
  //void CreateDevice(uint32_t a_deviceId);

  void BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView);

  void DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4 &a_wvp);

  void loadShaders();

  void SetupSimplePipeline();
  void RecreateSwapChain();

  void SetupDeviceExtensions();

  void AllocateResources();
  void PreparePipelines();

  void DestroyPipelines();
  void DeallocateResources();

  void InitPresentStuff();
  void ResetPresentStuff();
  void SetupGUIElements();
};


#endif// CHIMERA_SIMPLE_RENDER_H
