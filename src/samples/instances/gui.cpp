#include "instances_render.h"

#include "../../render/render_gui.h"

void InstancesRender::SetupGUIElements()
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Render();
}