#ifndef MODULES_COMMON_VISUALIZATION_OPENCV_EGO_RENDERER_H_
#define MODULES_COMMON_VISUALIZATION_OPENCV_EGO_RENDERER_H_

#include "modules/common/util/colormap.h"
#include "modules/common/visualization/common/proto/ego_render.pb.h"
#include "modules/common/visualization/opencv/canvas_transform.h"
#include "modules/common/visualization/opencv/layer_renderer.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace common {
namespace visualization {

class EgoRenderer
    : public LayerRendererT<::roadstar::localization::Localization> {
 public:
  explicit EgoRenderer(const EgoRenderParam &param = EgoRenderParam())
      : param_(param) {}

  virtual ~EgoRenderer() = default;

  void DoRender(const ::roadstar::localization::Localization &,
                const CanvasTransform &, cv::Mat *mat) override;

 private:
  const EgoRenderParam param_;
};

using TSEgoRenderer = TSLayerRenderer<EgoRenderer>;

void DrawEgoBox(const ::roadstar::localization::Localization &,
                const CanvasTransform &trans, cv::Mat *mat,
                double box_width = 2.2, double box_length = 6,
                ::roadstar::common::util::Color color =
                    ::roadstar::common::util::ColorName::Yellow,
                double line_width = 1.0);

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
