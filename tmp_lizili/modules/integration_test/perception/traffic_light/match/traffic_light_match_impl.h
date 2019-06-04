#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_TRFFIC_LIGHT_MATCH_TRAFFIC_LIGHT_MATCH_IMPL_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_TRFFIC_LIGHT_MATCH_TRAFFIC_LIGHT_MATCH_IMPL_H

#include <iostream>
#include <memory>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include "modules/common/proto/geometry.pb.h"
#include "modules/integration_test/perception/traffic_light/match/traffic_light_match_interface.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_detection_report_model.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"

namespace roadstar {
namespace integration_test {

namespace bg = boost::geometry;
using point_t = bg::model::point<double, 2, bg::cs::cartesian>;
using polygon_t = bg::model::polygon<point_t>;
using PointENU = roadstar::common::PointENU;  // actually the z value here isn't
                                              // useful for now.
class TrafficLightMatchImpl : public TrafficLightMatchInterface {
 public:
  explicit TrafficLightMatchImpl(const std::shared_ptr<ConfigModel>& config)
      : config_(config) {}

  TrafficLightDetectionReportModelPtr Match(
      const PerceptionTrafficLightDetectionModelPtrVec&,
      const LabeledTrafficLightDetectionModelPtrVec&) override;

 private:
  PerceptionTrafficLightDetectionModelPtr FindMatchFrame(
      const PerceptionTrafficLightDetectionModelPtrVec&, const double&,
      uint32_t* begin);
  int ComparePerFrame(const PerceptionTrafficLightDetectionModelPtr&,
                      const LabeledTrafficLightDetectionModelPtr&,
                      TrafficLightsFramePtr);
  bool IsMatch(const TrafficLightModel&, const TrafficLightModel&);
  polygon_t FillPolygon(const Eigen::Vector4d&);
  void FilterAccordingRoi(const LabeledTrafficLightDetectionModelPtrVec&,
                          const PerceptionTrafficLightDetectionModelPtrVec&);

 private:
  std::shared_ptr<ConfigModel> config_;
  LabeledTrafficLightDetectionModelPtrVec labeled_models_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
