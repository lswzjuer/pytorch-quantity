#include "modules/common/visualization/opencv/hdmap_renderer.h"

#include "opencv2/opencv.hpp"

#include "modules/common/visualization/opencv/geometry_drawing.h"

namespace roadstar {
namespace common {
namespace visualization {

void HdmapRenderer::DoRender(const HdmapViewBuilder& view,
                           const CanvasTransform& trans, cv::Mat* mat) {
  const auto& sections = view.sections_;
  const auto& connections = view.connections_;
  for (const auto& section : sections) {
    DrawSection(section, trans, mat, param_.lane_marker_color(),
                param_.lane_marker_width());
  }
  for (const auto& connection : connections) {
    DrawConnection(connection, trans, mat, param_.connection_color(),
                   param_.connection_width());
  }
}

//==============================================================================
//== Helper Functions ==========================================================
//==============================================================================

void DrawLaneMarker(const ::roadstar::hdmap::Lanemarker& lane_marker,
                    const CanvasTransform& trans, cv::Mat* mat,
                    ::roadstar::common::util::Color color, double width) {
  DrawCurve(lane_marker.curve(), trans, mat, color, width);
}

void DrawSection(const ::roadstar::hdmap::Section& section,
                 const CanvasTransform& trans, cv::Mat* mat,
                 ::roadstar::common::util::Color color, double width) {
  for (const auto& lanemarker : section.lanemarkers()) {
    DrawLaneMarker(lanemarker, trans, mat);
  }
}

void DrawConnection(const ::roadstar::hdmap::Connection& connection,
                    const CanvasTransform& trans, cv::Mat* mat,
                    ::roadstar::common::util::Color color, double width) {
  DrawPolygon(connection.polygon(), trans, mat, color, width);
}

}  // namespace visualization
}  // namespace common
}  // namespace roadstar
