#include "modules/integration_test/perception/traffic_light/match/traffic_light_match_impl.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "modules/common/log.h"
#include "modules/integration_test/perception/util/tool.h"

namespace roadstar {
namespace integration_test {

namespace {
constexpr double TRAFFIC_LIGHT_IOU = 0.3;
};

TrafficLightDetectionReportModelPtr TrafficLightMatchImpl::Match(
    const PerceptionTrafficLightDetectionModelPtrVec&
        perception_traffic_lights_data,
    const LabeledTrafficLightDetectionModelPtrVec&
        labeled_traffic_lights_data) {
  FilterAccordingRoi(labeled_traffic_lights_data,
                     perception_traffic_lights_data);
  TrafficLightDetectionReportModelPtr report(
      new TrafficLightDetectionReportModel);
  int total_match_frames = 0;
  uint32_t perception_it = 0;
  for (const auto& it : labeled_models_) {
    auto perception_lights = FindMatchFrame(perception_traffic_lights_data,
                                            it->GetTimestamp(), &perception_it);
    if (perception_lights) {
      ++total_match_frames;
      TrafficLightsFramePtr frame_ptr(new TrafficLightFrameModel);
      report->AddTrafficLightFramePtr(frame_ptr);
      ComparePerFrame(perception_lights, it, frame_ptr);
    } else {
      // search from begin
      perception_it = 0;
    }
  }
  report->SetTotalLabeledFrames(labeled_models_.size())
      ->SetTotalPerceptionFrames(perception_traffic_lights_data.size())
      ->SetTotalMatchFrames(total_match_frames)
      ->Calculate()
      ->ShowInfo();
  return report;
}

PerceptionTrafficLightDetectionModelPtr TrafficLightMatchImpl::FindMatchFrame(
    const PerceptionTrafficLightDetectionModelPtrVec& data,
    const double& timestamp, uint32_t* begin) {
  for (uint32_t it = *begin; it < data.size(); ++it) {
    double diff = timestamp - data[it]->GetTimestamp();
    if (diff > -0.01 && diff < 0.01) {
      *begin = it + 1;
      return data[it];
    }
  }
  return nullptr;
}

int TrafficLightMatchImpl::ComparePerFrame(
    const PerceptionTrafficLightDetectionModelPtr& perception_data,
    const LabeledTrafficLightDetectionModelPtr& labeled_data,
    TrafficLightsFramePtr frame_ptr) {
  int match_count = 0;
  std::size_t perception_data_size = perception_data->TrafficLightSize();
  std::size_t labeled_data_size = labeled_data->TrafficLightSize();
  std::set<std::size_t> match_set;
  for (std::size_t i = 0; i < perception_data_size; ++i) {
    const auto& perception_model = perception_data->GetTrafficLight(i);
    for (std::size_t j = 0; j < labeled_data_size; ++j) {
      if (match_set.count(j) == 1) {
        continue;
      }
      const auto& labeled_model = labeled_data->GetTrafficLight(j);
      if (IsMatch(perception_model, labeled_model)) {
        TrafficLightPtr labeled_model_ptr(new TrafficLightModel(labeled_model));
        TrafficLightPtr perception_model_ptr(
            new TrafficLightModel(perception_model));
        frame_ptr->AddTrafficLightPair(
            std::make_pair(labeled_model_ptr, perception_model_ptr));
        match_count++;
        match_set.insert(j);
      }
    }
  }
  frame_ptr->SetTimestamp(labeled_data->GetTimestamp())
      ->SetLabeledTotal(labeled_data_size)
      ->SetPerceptionTotal(perception_data_size)
      ->SetPerceptionCountdownLightsSize(
          perception_data->CountTimeTrafficLightSize())
      ->SetLabeledCountdownLightsSize(labeled_data->CountTimeTrafficLightSize())
      ->CalculateWithoutIgnore()
      ->ShowInfo();
  return match_count;
}

bool TrafficLightMatchImpl::IsMatch(const TrafficLightModel& perception_model,
                                    const TrafficLightModel& labeled_model) {
  polygon_t perception_pgn = FillPolygon(perception_model.GetBox4d());
  polygon_t labeled_pgn = FillPolygon(labeled_model.GetBox4d());
  std::vector<polygon_t> in, un;
  bg::intersection(labeled_pgn, perception_pgn, in);
  bg::union_(labeled_pgn, perception_pgn, un);
  if (in.empty()) {
    return false;
  }
  double inter_area = in.empty() ? 0 : bg::area(in.front());
  double union_area = bg::area(un.front());
  double overlap = inter_area / union_area;
  if (overlap < TRAFFIC_LIGHT_IOU) {
    return false;
  }
  bool is_equal =
      perception_model.GetColor() == labeled_model.GetColor() &&
      perception_model.GetLightType() == labeled_model.GetLightType();
  return is_equal;
}

polygon_t TrafficLightMatchImpl::FillPolygon(const Eigen::Vector4d& box4d) {
  polygon_t polygon;
  double xmin = box4d[0];
  double ymin = box4d[1];
  double xmax = box4d[2];
  double ymax = box4d[3];
  bg::append(polygon.outer(), point_t(xmin, ymin));
  bg::append(polygon.outer(), point_t(xmin, ymax));
  bg::append(polygon.outer(), point_t(xmax, ymax));
  bg::append(polygon.outer(), point_t(xmax, ymin));
  bg::append(polygon.outer(), point_t(xmin, ymin));
  return polygon;
}

void TrafficLightMatchImpl::FilterAccordingRoi(
    const LabeledTrafficLightDetectionModelPtrVec& labeled_data,
    const PerceptionTrafficLightDetectionModelPtrVec& perception_data) {
  uint32_t perception_it = 0;
  uint32_t total_filter = 0;
  for (const auto& it : labeled_data) {
    auto perception_frame =
        FindMatchFrame(perception_data, it->GetTimestamp(), &perception_it);
    if (perception_frame) {
      auto roi = perception_frame->GetRoi();
      polygon_t roi_pgn = FillPolygon(roi);
      std::size_t size = it->TrafficLightSize();
      LabeledTrafficLightDetectionModelPtr detection_model(
          new LabeledTrafficLightDetectionModel);
      detection_model->SetTimestamp(it->GetTimestamp());
      for (uint32_t index = 0; index < size; ++index) {
        const auto& light_moel = it->GetTrafficLight(index);
        polygon_t labeled_pgn = FillPolygon(light_moel.GetBox4d());
        std::vector<polygon_t> in;
        bg::intersection(labeled_pgn, roi_pgn, in);
        if (in.empty()) {
          continue;
        }
        double inter_area = in.empty() ? 0 : bg::area(in.front());
        if (inter_area > 0) {
          detection_model->AddTrafficLight(light_moel);
        }
      }
      total_filter += size - detection_model->TrafficLightSize();
      labeled_models_.push_back(detection_model);
    } else {
      perception_it = 0;
    }
  }
  AINFO << "TrafficLightMatchImpl::total filter = " << total_filter;
}

}  // namespace integration_test
}  // namespace roadstar
