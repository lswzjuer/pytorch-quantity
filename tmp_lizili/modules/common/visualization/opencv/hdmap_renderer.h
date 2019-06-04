#ifndef MODULES_COMMON_VISUALIZATION_OPENCV_HDMAP_RENDERER_H_
#define MODULES_COMMON_VISUALIZATION_OPENCV_HDMAP_RENDERER_H_

#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

#include "modules/common/util/colormap.h"
#include "modules/common/visualization/common/proto/hdmap_render.pb.h"
#include "modules/common/visualization/opencv/canvas_transform.h"
#include "modules/common/visualization/opencv/layer_renderer.h"
#include "modules/msgs/hdmap/proto/connection.pb.h"
#include "modules/msgs/hdmap/proto/hdmap_common.pb.h"
#include "modules/msgs/hdmap/proto/section.pb.h"

namespace roadstar {
namespace common {
namespace visualization {

class HdmapViewBuilder {
 public:
  void AddSection(const ::roadstar::hdmap::Section &section) {
    sections_.emplace_back(section);
  }
  void AddConnection(const ::roadstar::hdmap::Connection &connection) {
    connections_.emplace_back(connection);
  }
  void AddSection(::roadstar::hdmap::Section &&section) {
    sections_.emplace_back(std::move(section));
  }
  void AddConnection(::roadstar::hdmap::Connection &&connection) {
    connections_.emplace_back(std::move(connection));
  }

 private:
  friend class HdmapRenderer;
  std::vector<::roadstar::hdmap::Connection> connections_;
  std::vector<::roadstar::hdmap::Section> sections_;
};

class HdmapRenderer : public LayerRendererT<HdmapViewBuilder> {
 public:
  explicit HdmapRenderer(const HdmapRenderParam &param = HdmapRenderParam())
      : param_(param) {}

  virtual ~HdmapRenderer() = default;

  void DoRender(const HdmapViewBuilder &map_view, const CanvasTransform &trans,
                cv::Mat *mat) override;

 private:
  const HdmapRenderParam param_;
};

using TSHdmapRenderer = TSLayerRenderer<HdmapRenderer>;

//==============================================================================
//== Helpers ===================================================================
//==============================================================================

void DrawLaneMarker(const ::roadstar::hdmap::Lanemarker &lane_marker,
                    const CanvasTransform &trans, cv::Mat *mat,
                    ::roadstar::common::util::Color color =
                        ::roadstar::common::util::ColorName::Green,
                    double width = 1.0);

void DrawSection(const ::roadstar::hdmap::Section &section,
                 const CanvasTransform &trans, cv::Mat *mat,
                 ::roadstar::common::util::Color color =
                     ::roadstar::common::util::ColorName::Green,
                 double width = 1.0);

void DrawConnection(const ::roadstar::hdmap::Connection &connection,
                    const CanvasTransform &trans, cv::Mat *mat,
                    ::roadstar::common::util::Color color =
                        ::roadstar::common::util::ColorName::Blue,
                    double width = 1.0);

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
