#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>


/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment
  });

  mainViewColor = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_color",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
  });

  mainViewBlurredColor = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_blurred_color",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
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

  int windowSize = 11;

  blurCoeffs = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = windowSize * sizeof(float4),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "blur_coeffs",
  });
  
  float4* blurCoeffsData = reinterpret_cast<float4 *>(blurCoeffs.map());

  for (int i = 0; i < windowSize; ++i)
  {
    int dist = std::abs(i - windowSize / 2);
    blurCoeffsData[i] = float4(std::exp(-dist * dist / 2.0));
  }

  blurCoeffs.unmap();
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
  SetupSimplePipeline();
}

void SimpleShadowmapRender::loadShaders()
{
  etna::create_program("simple_material",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_shadow.frag.spv", VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("blur",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/blur.comp.spv"});
  etna::create_program("simple_shadow", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
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
  m_blurPipeline = pipelineManager.createComputePipeline("blur", {});
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

  pushConst2M.projView = a_wvp;
  for (uint32_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    auto inst         = m_pScnMgr->GetInstanceInfo(i);
    pushConst2M.model = m_pScnMgr->GetInstanceMatrix(i);
    vkCmdPushConstants(a_cmdBuff, m_basicForwardPipeline.getVkPipelineLayout(),
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
    etna::RenderTargetState renderTargets(a_cmdBuff, {2048, 2048}, {}, shadowMap);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
    DrawSceneCmd(a_cmdBuff, m_lightMatrix);
  }

  //// draw scene
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("simple_material");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, constants.genBinding()},
      etna::Binding {1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height}, {{mainViewColor.get(), mainViewColor.getView({})}}, mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicForwardPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_basicForwardPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj);
  }

  {
    etna::set_state(a_cmdBuff, mainViewColor.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderRead,
      vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(a_cmdBuff);
  }

  if (m_enableBlur)
  {
    //// blur
    //
    {
      auto blurInfo = etna::get_shader_program("blur");
    
      auto set = etna::create_descriptor_set(blurInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding {0, mainViewColor.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
        etna::Binding {1, mainViewBlurredColor.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
        etna::Binding {2, blurCoeffs.genBinding()}
      });
    
      VkDescriptorSet vkSet = set.getVkSet();
    
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_blurPipeline.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_blurPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);
    
      struct BlurParams
      {
        float variance;
      };
    
      BlurParams blurParams = {
        .variance = 1.0,
      };
    
      etna::flush_barriers(a_cmdBuff);
    
      vkCmdPushConstants(a_cmdBuff, m_blurPipeline.getVkPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurParams), &blurParams);
      vkCmdDispatch(a_cmdBuff, m_width, m_height, 1);
    }
  }

  if (m_enableBlur)
  {
    {
      etna::set_state(a_cmdBuff, a_targetImage,
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferWrite,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::set_state(a_cmdBuff, mainViewBlurredColor.get(),
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferRead,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(a_cmdBuff);
    }

    VkImageBlit blitRegion
    {
      .srcSubresource =
      {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .srcOffsets =
      {
        VkOffset3D{0, 0, 0},
        VkOffset3D{static_cast<int>(m_width), static_cast<int>(m_height), 1}
      },
      .dstSubresource =
      {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .dstOffsets =
      {
        VkOffset3D{0, 0, 0},
        VkOffset3D{static_cast<int>(m_width), static_cast<int>(m_height), 1}
      }
    };
    vkCmdBlitImage(a_cmdBuff, mainViewBlurredColor.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      a_targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_NEAREST);
  }
  else
  {
    {
      etna::set_state(a_cmdBuff, a_targetImage,
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferWrite,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::set_state(a_cmdBuff, mainViewColor.get(),
        vk::PipelineStageFlagBits2::eBlit,
        vk::AccessFlagBits2::eTransferRead,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(a_cmdBuff);
    }

    VkImageBlit blitRegion
    {
      .srcSubresource =
      {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .srcOffsets =
      {
        VkOffset3D{0, 0, 0},
        VkOffset3D{static_cast<int>(m_width), static_cast<int>(m_height), 1}
      },
      .dstSubresource =
      {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .dstOffsets =
      {
        VkOffset3D{0, 0, 0},
        VkOffset3D{static_cast<int>(m_width), static_cast<int>(m_height), 1}
      }
    };
    vkCmdBlitImage(a_cmdBuff, mainViewColor.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      a_targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_NEAREST);
  }

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}
