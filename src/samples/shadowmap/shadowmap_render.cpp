#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>

#include "../../../resources/shaders/light.h"


/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  m_mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment
  });

  m_albedoDepthMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "albedo_depth_map",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  m_normalMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "normal_map",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  m_defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  m_constants = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants"
  });

  m_uboMappedMem = m_constants.map();

  std::array<float3, 8> lightVolumeVertices = {
    float3(-1.0, -1.0, -1.0),
    float3(-1.0, -1.0,  1.0),
    float3(-1.0,  1.0, -1.0),
    float3(-1.0,  1.0,  1.0),
    float3( 1.0, -1.0, -1.0),
    float3( 1.0, -1.0,  1.0),
    float3( 1.0,  1.0, -1.0),
    float3( 1.0,  1.0,  1.0),
  };

  std::array<uint16_t, 36> lightVolumeIndices = {
    0, 1, 5, 0, 5, 4, // X-
    2, 6, 7, 2, 7, 3, // X+
    0, 2, 3, 0, 3, 1, // Y-
    4, 5, 7, 4, 7, 6, // Y+
    0, 4, 6, 0, 6, 2, // Z-
    1, 3, 7, 1, 7, 5, // Z+
  };
  
  m_lightVolumeVertexBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = std::span(lightVolumeVertices).size_bytes(),
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "light_volume_vertex_buffer",
  });

  m_lightVolumeIndexBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = std::span(lightVolumeIndices).size_bytes(),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "light_volume_index_buffer",
  });

  float3 *lightVolumeVertexBufferData = reinterpret_cast<float3 *>(m_lightVolumeVertexBuffer.map());
  std::copy(lightVolumeVertices.begin(), lightVolumeVertices.end(), lightVolumeVertexBufferData);
  m_lightVolumeVertexBuffer.unmap();

  uint16_t *lightVolumeIndexBufferData = reinterpret_cast<uint16_t *>(m_lightVolumeIndexBuffer.map());
  std::copy(lightVolumeIndices.begin(), lightVolumeIndices.end(), lightVolumeIndexBufferData);
  m_lightVolumeIndexBuffer.unmap();

  m_lightData = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(LightInfo) * m_lightCount,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "light_data",
  });

  LightInfo *lightData = reinterpret_cast<LightInfo *>(m_lightData.map());
  for (size_t i = 0; i < m_lightCount; ++i)
  {
    float a = 1.0;
    float b = 1.0;
    float c = 50.0;
    float threshold = 0.001;

    float radius = (std::sqrt(b * b - 4 * a * c + 4 * c / threshold) - b) / c / 2.0;

    float3 position = float3((i * 17 + 7) % 21 / 20.0, i * 29 % 31 / 30.0, (i * 13 + 2) % 24 / 23.0) * 2.0 - 1.0;
    float3 color = float3(i * 3 % 7 / 6.0, (i * 9 + 2) % 4 / 3.0, (i * 5 + 3) % 9 / 8.0);

    lightData[i] = LightInfo
    {
      .position = float4(position.x, position.y, position.z, 0.0),
      .color = float4(color.x, color.y, color.z, 0.0),
      .parameters = float4(a, b, c, radius),
    };
  }
  m_lightData.unmap();

  m_viewParamsBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(m_viewParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "view_params",
  });
  m_viewParamsBuffer.map();
}

void SimpleShadowmapRender::LoadScene(const char* path, bool transpose_inst_matrices)
{
  m_pScnMgr->LoadSceneXML(path, transpose_inst_matrices);

  // TODO: Make a separate stage
  loadShaders();
  PreparePipelines();

  auto loadedCam = m_pScnMgr->GetCamera(0);
  m_cam.fov = loadedCam.fov;
  m_cam.pos = float3(loadedCam.pos);
  m_cam.up  = float3(loadedCam.up);
  m_cam.lookAt = float3(loadedCam.lookAt);
  m_cam.tdist  = loadedCam.farPlane;
}

void SimpleShadowmapRender::DeallocateResources()
{
  m_mainViewDepth.reset(); // TODO: Make an etna method to reset all the resources
  m_albedoDepthMap.reset();
  m_normalMap.reset();;
  m_swapchain.Cleanup();
  vkDestroySurfaceKHR(GetVkInstance(), m_surface, nullptr);  

  m_constants = etna::Buffer();
}





/// PIPELINES CREATION

void SimpleShadowmapRender::PreparePipelines()
{
  SetupSimplePipeline();
}

void SimpleShadowmapRender::loadShaders()
{
  etna::create_program("g_buffer_pass",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/g_buffer_pass.vert.spv",
     VK_GRAPHICS_BASIC_ROOT"/resources/shaders/g_buffer_pass.frag.spv"});
  etna::create_program("deferred_shading_pass",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/deferred_shading_pass.vert.spv",
     VK_GRAPHICS_BASIC_ROOT"/resources/shaders/deferred_shading_pass.frag.spv"});
}

void SimpleShadowmapRender::SetupSimplePipeline()
{
  etna::VertexShaderInputDescription sceneVertexInputDesc
    {
      .bindings = {etna::VertexShaderInputDescription::Binding
        {
          .byteStreamDescription = m_pScnMgr->GetVertexStreamDescription()
        }}
    };

  etna::VertexShaderInputDescription lightVolumeVertexInputDesc
    {
      .bindings = {etna::VertexShaderInputDescription::Binding
        {
          .byteStreamDescription = etna::VertexByteStreamFormatDescription
            {
              sizeof(float3),
              {
                etna::VertexByteStreamFormatDescription::Attribute {vk::Format::eR32G32B32Sfloat, 0}
              }
            }
        }}
    };

  auto& pipelineManager = etna::get_context().getPipelineManager();
  m_gBufferPassPipeline = pipelineManager.createGraphicsPipeline("g_buffer_pass",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .blendingConfig =
        {
          .attachments{
            2,
            {
              .blendEnable = false,
              .colorWriteMask = vk::ColorComponentFlagBits::eR
                | vk::ColorComponentFlagBits::eG
                | vk::ColorComponentFlagBits::eB
                | vk::ColorComponentFlagBits::eA
            }}
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats =
          {
            vk::Format::eR32G32B32A32Sfloat,
            vk::Format::eR32G32B32A32Sfloat,
          },
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        },
    });
  m_deferredShadingPassPipeline = pipelineManager.createGraphicsPipeline("deferred_shading_pass",
    {
      .vertexShaderInput = lightVolumeVertexInputDesc,
      .rasterizationConfig =
        {
          .cullMode = vk::CullModeFlagBits::eFront,
          .frontFace = vk::FrontFace::eClockwise,
          .lineWidth = 1.0,
        },
      .blendingConfig =
        {
          .attachments =
            {
              vk::PipelineColorBlendAttachmentState{
                .blendEnable = true,
                .srcColorBlendFactor = vk::BlendFactor::eOne,
                .dstColorBlendFactor = vk::BlendFactor::eOne,
                .colorBlendOp = vk::BlendOp::eAdd,
                .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                .dstAlphaBlendFactor = vk::BlendFactor::eOne,
                .alphaBlendOp = vk::BlendOp::eAdd,
                .colorWriteMask = vk::ColorComponentFlagBits::eR
                  | vk::ColorComponentFlagBits::eG
                  | vk::ColorComponentFlagBits::eB
                  | vk::ColorComponentFlagBits::eA,
              }
            }
        },
      .depthConfig =
        {
          .depthTestEnable = false,
          .depthWriteEnable = false,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {static_cast<vk::Format>(m_swapchain.GetFormat())},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        }
    });
}

void SimpleShadowmapRender::DestroyPipelines()
{
}



/// COMMAND BUFFER FILLING

void SimpleShadowmapRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4& a_wvp)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT);

  VkDeviceSize zero_offset = 0u;
  VkBuffer vertexBuf = m_pScnMgr->GetVertexBuffer();
  VkBuffer indexBuf  = m_pScnMgr->GetIndexBuffer();
  
  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
  vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  for (uint32_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    auto inst         = m_pScnMgr->GetInstanceInfo(i);
    pushConst2M.model = m_pScnMgr->GetInstanceMatrix(i);
    vkCmdPushConstants(a_cmdBuff, m_gBufferPassPipeline.getVkPipelineLayout(),
      stageFlags, 0, sizeof(pushConst2M), &pushConst2M);

    auto mesh_info = m_pScnMgr->GetMeshInfo(inst.mesh_id);
    vkCmdDrawIndexed(a_cmdBuff, mesh_info.m_indNum, 1, mesh_info.m_indexOffset, mesh_info.m_vertexOffset, 0);
  }
}

void SimpleShadowmapRender::DrawLightVolumesCmd(VkCommandBuffer a_cmdBuff)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT);

  VkDeviceSize zero_offset = 0u;
  VkBuffer vertexBuf = m_lightVolumeVertexBuffer.get();

  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
  vkCmdBindIndexBuffer(a_cmdBuff, m_lightVolumeIndexBuffer.get(), 0, VK_INDEX_TYPE_UINT16);

  vkCmdPushConstants(a_cmdBuff, m_gBufferPassPipeline.getVkPipelineLayout(),
      stageFlags, 0, sizeof(pushConst2M), &pushConst2M);

  vkCmdDrawIndexed(a_cmdBuff, 36, m_lightCount, 0, 0, 0);
}

void SimpleShadowmapRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// draw scene to G-Buffer
  //
  {
    auto gBufferPassInfo = etna::get_shader_program("g_buffer_pass");

    auto set = etna::create_descriptor_set(gBufferPassInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, m_viewParamsBuffer.genBinding()},
    });

    VkDescriptorSet vkSet = set.getVkSet();

    vk::ImageView albedoMapView = m_albedoDepthMap.getView({});
    vk::ImageView normalMapView = m_normalMap.getView({});

    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height},
      {
        {m_albedoDepthMap.get(), albedoMapView},
        {m_normalMap.get(), normalMapView},
      },
      m_mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferPassPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_gBufferPassPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);
    
    DrawSceneCmd(a_cmdBuff, m_worldViewProj);
  }

  //// draw lights to screen
  //
  {
    auto deferredShadingPassInfo = etna::get_shader_program("deferred_shading_pass");

    auto set = etna::create_descriptor_set(deferredShadingPassInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, m_albedoDepthMap.genBinding(m_defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {1, m_normalMap.genBinding(m_defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {2, m_lightData.genBinding()},
      etna::Binding {3, m_viewParamsBuffer.genBinding()},
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height}, {{a_targetImage, a_targetImageView}}, m_mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_deferredShadingPassPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_deferredShadingPassPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawLightVolumesCmd(a_cmdBuff);
  }

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}
