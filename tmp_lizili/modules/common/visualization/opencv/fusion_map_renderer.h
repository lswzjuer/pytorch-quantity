#ifndef MODULES_COMMON_VISUALIZATION_OPENCV_FUSIONMAP_RENDERER_H_
#define MODULES_COMMON_VISUALIZATION_OPENCV_FUSIONMAP_RENDERER_H_

#include "modules/common/util/colormap.h"
#include "modules/common/visualization/common/proto/fusion_map_render.pb.h"
#include "modules/common/visualization/opencv/canvas_transform.h"
#include "modules/common/visualization/opencv/layer_renderer.h"
#include "modules/msgs/perception/proto/fusion_map.pb.h"

namespace roadstar {
namespace common {
namespace visualization {

class FusionMapRenderer
    : public LayerRendererT<::roadstar::perception::FusionMap> {
 public:
  explicit FusionMapRenderer(
      const FusionMapRenderParam &param = FusionMapRenderParam())
      : param_(param) {}

  virtual ~FusionMapRenderer() = default;

  void DoRender(const ::roadstar::perception::FusionMap &fusion_map,
                const CanvasTransform &trans, cv::Mat *mat) override;

 private:
  const FusionMapRenderParam param_;
};

using TSFusionMapRenderer = TSLayerRenderer<FusionMapRenderer>;

void DrawObstacleBox(const ::roadstar::perception::Obstacle &obstacle,
                     const CanvasTransform &trans, cv::Mat *mat,
                     ::roadstar::common::util::Color color =
                         ::roadstar::common::util::ColorName::Red,
                     double width = 1.0);

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
