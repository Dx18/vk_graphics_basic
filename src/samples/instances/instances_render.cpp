#include "instances_render.h"

#include <random>

#include <etna/RenderTargetStates.hpp>

void InstancesRender::AllocateResources()
{
  m_mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });

  m_defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});

  m_matrixBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(float4x4) * m_instanceCount,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "matrices"
  });
  
  {
    float4x4 *matrixBuffer = reinterpret_cast<float4x4 *>(m_matrixBuffer.map());

    std::mt19937 random(std::random_device{}());
    std::uniform_real_distribution<float> genCoordinate(0.0, 30.0);
    std::uniform_real_distribution<float> genHeight(0.0, 5.0);

    std::vector<float3> hills;
    for (uint32_t j = 0; j < 14; ++j)
    {
      hills.push_back(float3(genCoordinate(random), genHeight(random), genCoordinate(random)));
    }


    for (uint32_t i = 0; i < m_instanceCount; ++i)
    {
      float x = genCoordinate(random);
      float z = genCoordinate(random);
      
      float y = 0.0;
      for (uint32_t j = 0; j < hills.size(); ++j)
      {
        float2 displacement = float2(x, z) - float2(hills[j].x, hills[j].z);
        y += 1.0 / (1.0 + 2.0 * dot(displacement, displacement));
      }
     
      matrixBuffer[i] = translate4x4(float3(x, y, z)) * scale4x4(float3(0.01, 0.01, 0.01));
    }
    m_matrixBuffer.unmap();
  }

  m_indexBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(uint32_t) * m_instanceCount,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "indices"
  });
  m_drawCommandBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(VkDrawIndexedIndirectCommand),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "draw_command"
  });
  m_drawCommandBuffer.map();
}

void InstancesRender::LoadScene(const char *path, bool transpose_inst_matrices)
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

void InstancesRender::DeallocateResources()
{
  m_mainViewDepth.reset();
  m_swapchain.Cleanup();
  vkDestroySurfaceKHR(GetVkInstance(), m_surface, nullptr);

  m_matrixBuffer = etna::Buffer();
  m_indexBuffer = etna::Buffer();
  m_drawCommandBuffer = etna::Buffer();
}

void InstancesRender::PreparePipelines()
{
  SetupSimplePipeline();
}

void InstancesRender::loadShaders()
{
  etna::create_program("frustum_culling",
    { VK_GRAPHICS_BASIC_ROOT "/resources/shaders/frustum_culling.comp.spv" });
  etna::create_program("simple_material",
    { VK_GRAPHICS_BASIC_ROOT "/resources/shaders/instances.vert.spv", VK_GRAPHICS_BASIC_ROOT "/resources/shaders/instances.frag.spv" });
}

void InstancesRender::SetupSimplePipeline()
{
  std::vector<std::pair<VkDescriptorType, uint32_t>> dtypes = {
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }
  };
  m_pFrustumCullingBindings = std::make_shared<vk_utils::DescriptorMaker>(m_context->getDevice(), dtypes, 1);

  m_pFrustumCullingBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  m_pFrustumCullingBindings->BindBuffer(0, m_matrixBuffer.get());
  m_pFrustumCullingBindings->BindBuffer(1, m_indexBuffer.get());
  m_pFrustumCullingBindings->BindBuffer(2, m_drawCommandBuffer.get());
  m_pFrustumCullingBindings->BindEnd(&m_frustumCullingDS, &m_frustumCullingDSLayout);

  dtypes = {
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }
  };
  m_pSimpleMaterialBindings = std::make_shared<vk_utils::DescriptorMaker>(m_context->getDevice(), dtypes, 1);

  m_pSimpleMaterialBindings->BindBegin(VK_SHADER_STAGE_VERTEX_BIT);
  m_pSimpleMaterialBindings->BindBuffer(0, m_matrixBuffer.get());
  m_pSimpleMaterialBindings->BindBuffer(1, m_indexBuffer.get());
  m_pSimpleMaterialBindings->BindEnd(&m_simpleMaterialDS, &m_simpleMaterialDSLayout);

  auto &pipelineManager  = etna::get_context().getPipelineManager();

  m_frustumCullingPipeline = pipelineManager.createComputePipeline("frustum_culling",
    etna::ComputePipeline::CreateInfo{});

  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = { etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = m_pScnMgr->GetVertexStreamDescription() } }
  };
  m_simpleMaterialPipeline = pipelineManager.createGraphicsPipeline("simple_material",
    { .vertexShaderInput    = sceneVertexInputDesc,
      .fragmentShaderOutput = {
        .colorAttachmentFormats = { static_cast<vk::Format>(m_swapchain.GetFormat()) },
        .depthAttachmentFormat  = vk::Format::eD32Sfloat } });
}

void InstancesRender::DestroyPipelines()
{
  m_frustumCullingPipeline = etna::ComputePipeline();
  m_simpleMaterialPipeline = etna::GraphicsPipeline();
}

void InstancesRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4 &a_wvp)
{
  if (m_pScnMgr->InstancesNum() <= 0) {
    return;
  }

  VkDeviceSize zeroOffset = 0;
  VkBuffer vertexBuffer = m_pScnMgr->GetVertexBuffer();
  VkBuffer indexBuffer = m_pScnMgr->GetIndexBuffer();

  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuffer, &zeroOffset);
  vkCmdBindIndexBuffer(a_cmdBuff, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

  auto instance = m_pScnMgr->GetInstanceInfo(0);
  m_renderParams.mViewProjection = a_wvp;
  vkCmdPushConstants(a_cmdBuff, m_simpleMaterialPipeline.getVkPipelineLayout(),
    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(m_renderParams), &m_renderParams);

  auto meshInfo = m_pScnMgr->GetMeshInfo(instance.mesh_id);
  vkCmdDrawIndirect(a_cmdBuff, m_drawCommandBuffer.get(), 0, 1, 0);
}

void InstancesRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// perform frustum culling
  //
  {
    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_frustumCullingPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_frustumCullingPipeline.getVkPipelineLayout(), 0, 1, &m_frustumCullingDS, 0, NULL);

    vkCmdPushConstants(a_cmdBuff, m_frustumCullingPipeline.getVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_frustumCullingParams), &m_frustumCullingParams);

    vkCmdDispatch(a_cmdBuff, m_instanceCount, 1, 1);
  }

  //// draw final scene to screen
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("simple_material");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff, {
      etna::Binding{ 0, m_matrixBuffer.genBinding() },
      etna::Binding{ 1, m_indexBuffer.genBinding() },
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, { m_width, m_height }, { { a_targetImage, a_targetImageView } }, m_mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_simpleMaterialPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_simpleMaterialPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, NULL);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj);
  }

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe, vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR, vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}