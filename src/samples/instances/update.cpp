#include "../../utils/input_definitions.h"

#include "etna/Etna.hpp"
#include "instances_render.h"

void InstancesRender::UpdateCamera(const Camera *cams, uint32_t a_camsNumber)
{
  m_cam = cams[0];
  UpdateView();
}

void InstancesRender::UpdateView()
{
  ///// calc camera matrix
  //
  const float aspect  = float(m_width) / float(m_height);
  auto mProjFix = OpenglToVulkanProjectionMatrixFix();
  auto mProj = projectionMatrix(m_cam.fov, aspect, 0.1f, 1000.0f);
  auto mLookAt = LiteMath::lookAt(m_cam.pos, m_cam.lookAt, m_cam.up);
  auto mWorldViewProj = mProjFix * mProj * mLookAt;

  m_worldViewProj = mWorldViewProj;

  /////// calc light matrix
  ////
  //if (m_light.usePerspectiveM)
  //  mProj = perspectiveMatrix(m_light.cam.fov, 1.0f, 1.0f, m_light.lightTargetDist * 2.0f);
  //else
  //  mProj = ortoMatrix(-m_light.radius, +m_light.radius, -m_light.radius, +m_light.radius, 0.0f, m_light.lightTargetDist);

  //if (m_light.usePerspectiveM)// don't understang why fix is not needed for perspective case for shadowmap ... it works for common rendering
  //  mProjFix = LiteMath::float4x4();
  //else
  //  mProjFix = OpenglToVulkanProjectionMatrixFix();

  //mLookAt       = LiteMath::lookAt(m_light.cam.pos, m_light.cam.pos + m_light.cam.forward() * 10.0f, m_light.cam.up);
  //m_lightMatrix = mProjFix * mProj * mLookAt;
}

//void InstancesRender::UpdateUniformBuffer(float a_time)
//{
//}

void InstancesRender::ProcessInput(const AppInput &input)
{
  // add keyboard controls here
  // camera movement is processed separately
  //
  /*if (input.keyReleased[GLFW_KEY_Q])
    m_input.drawFSQuad = !m_input.drawFSQuad;

  if (input.keyReleased[GLFW_KEY_P])
    m_light.usePerspectiveM = !m_light.usePerspectiveM;*/

  // recreate pipeline to reload shaders
  if (input.keyPressed[GLFW_KEY_B])
  {
#ifdef WIN32
    std::system("cd ../resources/shaders && python compile_instances_shaders.py");
#else
    std::system("cd ../resources/shaders && python3 compile_instances_shaders.py");
#endif

    etna::reload_shaders();

    for (uint32_t i = 0; i < m_framesInFlight; ++i)
    {
      BuildCommandBufferSimple(m_cmdBuffersDrawMain[i], m_swapchain.GetAttachment(i).image, m_swapchain.GetAttachment(i).view);
    }
  }
}
