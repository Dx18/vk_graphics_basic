#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>
#include <random>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>


/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  mainView = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc
  });
  mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
  });
  ssaoOverlay = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "ssao_overlay",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage
  });
  ssaoResult = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "ssao_result",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc
  });

  shadowMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
  });

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  constants = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants"
  });

  m_uboMappedMem = constants.map();

  ssaoNoise = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(float4) * 128,
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "ssao_noise"
  });

  float *ssaoNoiseData = reinterpret_cast<float *>(ssaoNoise.map());

  std::mt19937 random;
  std::normal_distribution<float> genOffsetComponent(0.0, 1.0);
  std::uniform_real_distribution<float> genSampleDistance(0.0, 1.0);
  
  for (size_t i = 0; i < 64; ++i)
  {
    float3 offset;
    for (size_t j = 0; j < 3; ++j)
    {
      offset[j] = genOffsetComponent(random);
    }
    offset = LiteMath::normalize(offset);

    float distance = genSampleDistance(random);
    distance *= distance;

    offset *= distance;

    ssaoNoiseData[4 * i]     = offset.x;
    ssaoNoiseData[4 * i + 1] = offset.y;
    ssaoNoiseData[4 * i + 2] = offset.z;
    ssaoNoiseData[4 * i + 3] = 1.0 - distance;
  }

  std::uniform_real_distribution<float> genRotation(0.0, 1.0);

  for (size_t i = 0; i < 256; ++i)
  {
    ssaoNoiseData[256 + i] = genRotation(random) * 2 * M_PI;
  }

  ssaoNoise.unmap();
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
  mainViewDepth.reset(); // TODO: Make an etna method to reset all the resources
  shadowMap.reset();
  m_swapchain.Cleanup();
  vkDestroySurfaceKHR(GetVkInstance(), m_surface, nullptr);  

  constants = etna::Buffer();
}





/// PIPELINES CREATION

void SimpleShadowmapRender::PreparePipelines()
{
  // create full screen quad for debug purposes
  // 
  m_pQuad = std::make_unique<QuadRenderer>(QuadRenderer::CreateInfo{ 
      .format = static_cast<vk::Format>(m_swapchain.GetFormat()),
      .rect = { 0, 0, 512, 512 }, 
    });
  SetupSimplePipeline();
}

void SimpleShadowmapRender::loadShaders()
{
  etna::create_program("simple_material",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_shadow.frag.spv", VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("simple_shadow", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});

  etna::create_program("ssao_overlay", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/ssao_overlay.comp.spv"});
  etna::create_program("ssao_apply", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/ssao_apply.comp.spv"});
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

  auto& pipelineManager = etna::get_context().getPipelineManager();
  m_basicForwardPipeline = pipelineManager.createGraphicsPipeline("simple_material",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {vk::Format::eR32G32B32A32Sfloat},
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        }
    });
  m_shadowPipeline = pipelineManager.createGraphicsPipeline("simple_shadow",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm
        }
    });
  m_ssaoOverlayPipeline = pipelineManager.createComputePipeline("ssao_overlay", {});
  m_ssaoApplyPipeline = pipelineManager.createComputePipeline("ssao_apply", {});
}


/// COMMAND BUFFER FILLING

void SimpleShadowmapRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4& a_wvp, VkPipelineLayout a_pipelineLayout)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT);

  VkDeviceSize zero_offset = 0u;
  VkBuffer vertexBuf = m_pScnMgr->GetVertexBuffer();
  VkBuffer indexBuf  = m_pScnMgr->GetIndexBuffer();
  
  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
  vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  pushConst2M.projView = a_wvp;
  for (uint32_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    auto inst         = m_pScnMgr->GetInstanceInfo(i);
    pushConst2M.model = m_pScnMgr->GetInstanceMatrix(i);
    vkCmdPushConstants(a_cmdBuff, a_pipelineLayout,
      stageFlags, 0, sizeof(pushConst2M), &pushConst2M);

    auto mesh_info = m_pScnMgr->GetMeshInfo(inst.mesh_id);
    vkCmdDrawIndexed(a_cmdBuff, mesh_info.m_indNum, 1, mesh_info.m_indexOffset, mesh_info.m_vertexOffset, 0);
  }
}

void SimpleShadowmapRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// draw scene to shadowmap
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, 2048, 2048}, {}, {.image = shadowMap.get(), .view = shadowMap.getView({})});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
    DrawSceneCmd(a_cmdBuff, m_lightMatrix, m_shadowPipeline.getVkPipelineLayout());
  }

  //// draw final scene to a texture
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("simple_material");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, constants.genBinding()},
      etna::Binding {1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, m_width, m_height},
      {{.image = mainView.get(), .view = mainView.getView({})}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicForwardPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_basicForwardPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj, m_basicForwardPipeline.getVkPipelineLayout());
  }

  if (m_enableSSAO)
  {
    {
      etna::set_state(a_cmdBuff, mainViewDepth.get(), vk::PipelineStageFlagBits2::eComputeShader,
          vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eDepth);
      etna::set_state(a_cmdBuff, ssaoOverlay.get(), vk::PipelineStageFlagBits2::eComputeShader,
          vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(a_cmdBuff);
    
      auto ssaoOverlayInfo = etna::get_shader_program("ssao_overlay");
    
      auto set = etna::create_descriptor_set(ssaoOverlayInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding {0, mainViewDepth.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding {1, ssaoOverlay.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
        etna::Binding {2, ssaoNoise.genBinding()}
      });
    
      VkDescriptorSet vkSet = set.getVkSet();
    
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoOverlayPipeline.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoOverlayPipeline.getVkPipelineLayout(),
          0, 1, &vkSet, 0, VK_NULL_HANDLE);
    
      pushConst2M.projView = m_worldViewProj;
      vkCmdPushConstants(a_cmdBuff, m_ssaoOverlayPipeline.getVkPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConst2M), &pushConst2M);
    
      vkCmdDispatch(a_cmdBuff, m_width, m_height, 1);
    }
    
    {
      etna::set_state(a_cmdBuff, mainView.get(), vk::PipelineStageFlagBits2::eComputeShader,
          vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
      etna::set_state(a_cmdBuff, ssaoOverlay.get(), vk::PipelineStageFlagBits2::eComputeShader,
          vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
      etna::set_state(a_cmdBuff, ssaoResult.get(), vk::PipelineStageFlagBits2::eComputeShader,
          vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(a_cmdBuff);
    
      auto ssaoApplyInfo = etna::get_shader_program("ssao_apply");
    
      auto set = etna::create_descriptor_set(ssaoApplyInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding {0, mainView.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding {1, mainViewDepth.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding {2, ssaoOverlay.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding {3, ssaoResult.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)}
      });
    
      VkDescriptorSet vkSet = set.getVkSet();
    
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoApplyPipeline.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoApplyPipeline.getVkPipelineLayout(),
          0, 1, &vkSet, 0, VK_NULL_HANDLE);
    
      vkCmdDispatch(a_cmdBuff, m_width, m_height, 1);
    }
    
    {
      etna::set_state(a_cmdBuff, ssaoResult.get(), vk::PipelineStageFlagBits2::eTransfer,
          vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageAspectFlagBits::eColor);
      etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eTransfer,
          vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal, vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(a_cmdBuff);
    
      vk::ImageBlit region =
      {
        .srcSubresource =
          {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
          },
        .srcOffsets = std::array<vk::Offset3D, 2>
          {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int32_t>(m_width), static_cast<int32_t>(m_height), 1}
          },
        .dstSubresource =
          {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
          },
        .dstOffsets = std::array<vk::Offset3D, 2>
          {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int32_t>(m_width), static_cast<int32_t>(m_height), 1}
          },
      };
    
      VkImageBlit vkRegion = region;
    
      vkCmdBlitImage(a_cmdBuff, ssaoResult.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          a_targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &vkRegion, VK_FILTER_NEAREST);
    }
  }
  else
  {
    etna::set_state(a_cmdBuff, mainView.get(), vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageAspectFlagBits::eColor);
    etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal, vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(a_cmdBuff);
    
    vk::ImageBlit region =
    {
      .srcSubresource =
        {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = 0,
          .baseArrayLayer = 0,
          .layerCount = 1
        },
      .srcOffsets = std::array<vk::Offset3D, 2>
        {
          vk::Offset3D{0, 0, 0},
          vk::Offset3D{static_cast<int32_t>(m_width), static_cast<int32_t>(m_height), 1}
        },
      .dstSubresource =
        {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = 0,
          .baseArrayLayer = 0,
          .layerCount = 1
        },
      .dstOffsets = std::array<vk::Offset3D, 2>
        {
          vk::Offset3D{0, 0, 0},
          vk::Offset3D{static_cast<int32_t>(m_width), static_cast<int32_t>(m_height), 1}
        },
    };
    
    VkImageBlit vkRegion = region;
    
    vkCmdBlitImage(a_cmdBuff, mainView.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        a_targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &vkRegion, VK_FILTER_NEAREST);
  }

  if(m_input.drawFSQuad)
    m_pQuad->RecordCommands(a_cmdBuff, a_targetImage, a_targetImageView, shadowMap, defaultSampler);

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}
