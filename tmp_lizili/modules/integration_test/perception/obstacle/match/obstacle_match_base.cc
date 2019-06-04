#include "modules/integration_test/perception/obstacle/match/obstacle_match_base.h"

#include <algorithm>

#include "modules/common/log.h"
#include "modules/common/util/file.h"
#include "modules/integration_test/perception/util/visualization.h"
#include "modules/msgs/perception/proto/obstacle.pb.h"

namespace roadstar {
namespace integration_test {

using Localization = roadstar::localization::Localization;

template <typename T>
std::string GetVecInfo(std::vector<T> vec) {
  std::stringstream ss;
  for (auto x : vec) {
    ss << x << " ";
  }
  return ss.str();
}

void ObstacleMatchBase::CalculateLabeledObstacleVelocity(
    std::map<int, PerceptionFrameModel>::const_iterator prev_perception_it,
    std::map<int, LabelFrameModel>::const_iterator prev_label_it,
    std::map<int, PerceptionFrameModel>::const_iterator next_perception_it,
    std::map<int, LabelFrameModel>::const_iterator next_label_it,
    std::map<int, LabelFrameModel>::iterator current_label_it,
    size_t obstacle_id) {
  std::string version = config_->GetValueViaKey("perception");
  if (prev_perception_it == perception_obstacles_.end()) {
    AFATAL
        << "invalid prev_perception_it in CalculateLabeledObstacleVelocity()";
  }
  if (next_perception_it == perception_obstacles_.end()) {
    AFATAL
        << "invalid next_perception_it in CalculateLabeledObstacleVelocity()";
  }
  if (prev_perception_it == next_perception_it) {
    AWARN << "next_perception_it == prev_perception it in "
             "CalculateLabeledObstacleVelocity(); perception index is "
          << next_perception_it->first;
    return;
  }
  auto prev_location_model =
      const_cast<PerceptionFrameModel *>(&prev_perception_it->second)
          ->GetLocationModel();
  auto next_location_model =
      const_cast<PerceptionFrameModel *>(&next_perception_it->second)
          ->GetLocationModel();

  double prev_time = prev_location_model->GetTimeStamp();
  double next_time = next_location_model->GetTimeStamp();
  double duration = next_time - prev_time;

  auto current_obstacle_ego_front =
      current_label_it->second.GetEgoFrontTypeModelByID(obstacle_id);
  auto current_obstacle_velodyne =
      current_label_it->second.GetVelodyneTypeModelByID(obstacle_id);
  auto prev_obstacle_utm = const_cast<LabelFrameModel *>(&prev_label_it->second)
                               ->GetVelodyneTypeModelByID(obstacle_id);
  auto next_obstacle_utm = const_cast<LabelFrameModel *>(&next_label_it->second)
                               ->GetVelodyneTypeModelByID(obstacle_id);
  if (!prev_obstacle_utm || !current_obstacle_velodyne || !next_obstacle_utm ||
      !current_obstacle_ego_front) {
    AFATAL << "can't find obstacle id " << obstacle_id << " in frame "
           << current_label_it->first << " or " << prev_label_it->first
           << " or " << next_label_it->first;
  }

  auto prev_obstacle_utm_model =
      prev_obstacle_utm->Velodyne64ToUtm(version, *prev_location_model);
  auto next_obstacle_utm_model =
      next_obstacle_utm->Velodyne64ToUtm(version, *next_location_model);

  double distance = sqrt(
      pow(prev_obstacle_utm_model.GetX() - next_obstacle_utm_model.GetX(), 2) +
      pow(prev_obstacle_utm_model.GetY() - next_obstacle_utm_model.GetY(), 2) +
      pow(prev_obstacle_utm_model.GetZ() - next_obstacle_utm_model.GetZ(), 2));
  double velocity = distance / duration;
  double heading =
      atan2(next_obstacle_utm_model.GetY() - prev_obstacle_utm_model.GetY(),
            next_obstacle_utm_model.GetX() - prev_obstacle_utm_model.GetX());
  ADEBUG << "v:" << velocity << " d:" << distance << " t:" << duration;

  current_obstacle_ego_front->SetVelocity(velocity);
  current_obstacle_ego_front->SetHeading(heading);
  current_obstacle_velodyne->SetVelocity(velocity);
  current_obstacle_velodyne->SetHeading(heading);
}

void ObstacleMatchBase::ComputeLabeledObstacleVelocity(
    const std::map<int, LabelFrameModel> &labeled_frames,
    const std::map<int, PerceptionFrameModel> &perception_frames) {
  AINFO << "Begin ComputeLabeledObstacleVelocity...";
  AINFO << "before filter. perceptipn obstatcle frames = "
        << perception_frames.size()
        << " labeled obstacle frames = " << labeled_frames.size();
  FilterPerceptionObstacle(perception_frames);
  FilterLabelObstacle(labeled_frames);
  OutputLostPerceptionFramesInfo();
  AINFO << "After filter. perceptipn obstatcle frames = "
        << perception_obstacles_.size()
        << " labeled obstacle frames = " << labeled_obstacles_.size();

  if (perception_obstacles_.size() <= 1) {
    AWARN << "only one perception frame after filter, stop velocity computing";
    return;
  }

  double time_delta = 0.5;

  auto perception_frame_it = perception_obstacles_.begin();
  while (perception_frame_it != perception_obstacles_.end()) {
    int frame_index = perception_frame_it->first;
    auto label_frame_it = labeled_obstacles_.find(frame_index);
    if (label_frame_it == labeled_obstacles_.end()) {
      AWARN << "no corresponding label frame of index " << frame_index;
      perception_frame_it++;
      continue;
    }

    decltype(perception_frame_it) prev_perception_frame_it;
    decltype(perception_frame_it) next_perception_frame_it;
    decltype(label_frame_it) prev_label_frame_it;
    decltype(label_frame_it) next_label_frame_it;

    double current_time =
        perception_frame_it->second.GetLocationModel()->GetTimeStamp();

    for (size_t obstacle_index = 0;
         obstacle_index < label_frame_it->second.Size(); obstacle_index++) {
      prev_perception_frame_it = perception_frame_it;
      next_perception_frame_it = perception_frame_it;
      prev_label_frame_it = label_frame_it;
      next_label_frame_it = label_frame_it;
      int obstacle_id =
          label_frame_it->second.GetEgoFrontTypeModelAt(obstacle_index)
              ->GetId();

      double prev_time_duration = 0;
      double next_time_duration = 0;
      decltype(prev_perception_frame_it->second
                   .GetLocationModel()) prev_perception_location;
      decltype(next_perception_frame_it->second
                   .GetLocationModel()) next_perception_location;

      bool recover = false;
      while (prev_perception_frame_it != perception_obstacles_.begin()) {
        prev_perception_frame_it--;
        prev_label_frame_it =
            labeled_obstacles_.find(prev_perception_frame_it->first);
        if (prev_label_frame_it == labeled_obstacles_.end()) {
          AWARN << "No labeled frame of index "
                << prev_perception_frame_it->first;
          recover = true;
        } else {
          bool has_prev_obstacle =
              prev_label_frame_it->second.HasObstacleId(obstacle_id);
          if (!has_prev_obstacle) {
            recover = true;
          }
        }

        if (recover) {
          prev_perception_frame_it++;
          prev_perception_location =
              prev_perception_frame_it->second.GetLocationModel();
          double prev_time = prev_perception_location->GetTimeStamp();
          prev_time_duration = current_time - prev_time;
          auto prev_frame_index = prev_perception_frame_it->first;
          prev_label_frame_it = labeled_obstacles_.find(prev_frame_index);
          break;
        }

        prev_perception_location =
            prev_perception_frame_it->second.GetLocationModel();
        double prev_time = prev_perception_location->GetTimeStamp();
        prev_time_duration = current_time - prev_time;
        if (prev_time_duration >= time_delta) break;
      }

      recover = false;
      while (next_perception_frame_it != perception_obstacles_.end()) {
        next_label_frame_it =
            labeled_obstacles_.find(next_perception_frame_it->first);
        if (next_label_frame_it == labeled_obstacles_.end()) {
          AWARN << "No labeled frame of index "
                << next_perception_frame_it->first;
          recover = true;
        } else {
          bool has_next_obstacle =
              next_label_frame_it->second.HasObstacleId(obstacle_id);
          if (!has_next_obstacle) {
            recover = true;
          }
        }

        if (recover) {
          next_perception_frame_it--;
          auto next_frame_index = next_perception_frame_it->first;
          next_label_frame_it = labeled_obstacles_.find(next_frame_index);
          break;
        }

        next_perception_location =
            next_perception_frame_it->second.GetLocationModel();
        double next_time = next_perception_location->GetTimeStamp();
        next_time_duration = next_time - current_time;
        if (next_time_duration + prev_time_duration >= time_delta * 2) break;
        next_perception_frame_it++;
      }
      if (next_perception_frame_it == perception_obstacles_.end()) {
        next_perception_frame_it--;
      }

      CalculateLabeledObstacleVelocity(
          prev_perception_frame_it, prev_label_frame_it,
          next_perception_frame_it, next_label_frame_it, label_frame_it,
          obstacle_id);
      ADEBUG << "obstacle id " << obstacle_id << " prev_frame_index "
             << prev_perception_frame_it->first << " current_frame_index "
             << perception_frame_it->first << " next_frame_index "
             << next_perception_frame_it->first;
    }
    perception_frame_it++;
  }
}

template <typename T>
T ObstacleMatchBase::GetPn(const std::vector<T> &values, unsigned int n) {
  if (n > 100) {
    AFATAL << "n should not be greater than 100 as second parameter of GetPn()";
  }

  if (n == 0) return values[values.size() - 1];
  size_t index = (100 - n) * values.size() / 100;
  if (values.size() == 0) return T();
  if (index < 0 || index >= values.size()) {
    AFATAL << "invalid index " << index;
  }
  return values[index];
}

// similarity varies from -1 to 1 (1 means perfect match)
double ObstacleMatchBase::CalculateVelocitySimilarity(double v1, double v2,
                                                      double heading1,
                                                      double heading2) {
  v1 += 0.1;
  v2 += 0.1;
  double slow_v, quick_v, slow_heading, quick_heading;
  if (v1 < v2) {
    slow_v = v1;
    slow_heading = heading1;
    quick_v = v2;
    quick_heading = heading2;
  } else {
    slow_v = v2;
    slow_heading = heading2;
    quick_v = v1;
    quick_heading = heading1;
  }

  double slow_x = slow_v * cos(slow_heading);
  double slow_y = slow_v * sin(slow_heading);
  double quick_x = quick_v * cos(quick_heading);
  double quick_y = quick_v * sin(quick_heading);

  double similarity =
      (slow_x * quick_x + slow_y * quick_y) / (quick_v * quick_v);
  return similarity;
}

double ObstacleMatchBase::CalculateVelocityDiffNorm(double v1, double v2,
                                                    double heading1,
                                                    double heading2) {
  double x1 = v1 * cos(heading1);
  double y1 = v1 * sin(heading1);
  double x2 = v2 * cos(heading2);
  double y2 = v2 * sin(heading2);

  double norm = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
  return norm;
}

std::shared_ptr<ReportModel> ObstacleMatchBase::Match(
    const std::map<int, LabelFrameModel> &labeled_obstacles,
    const std::map<int, PerceptionFrameModel> &perception_obstacles) {
  std::shared_ptr<ReportModel> report_model = std::make_shared<ReportModel>();

  int frames = 0;
  int matches_total = 0;
  int perception_obs_total = 0;
  int labeled_obs_total = 0;
  int zero_ob_frame_count = 0;
  std::vector<double> velocity_sim_all_obstacles;
  std::vector<double> velocity_diff_norm_whole;
  std::string png_path = config_->GetValueViaKey("mid_file_path") + "/pngs";
  std::string version = config_->GetValueViaKey("perception");
  std::string drive_scene = config_->GetValueViaKey("drive_scene");
  report_model->SetDriveScene(drive_scene);
  roadstar::common::util::EnsureDirectory(png_path);
  for (auto it = perception_obstacles_.begin();
       it != perception_obstacles_.end(); ++it) {
    const auto label_it = labeled_obstacles_.find(it->first);
    if (label_it != labeled_obstacles_.end()) {
      frames++;
      LabelFrameModel &label_frame_model = label_it->second;
      PerceptionFrameModel &perception_frame_model = it->second;
      Visualization vs;
      vs.DrawFramePng(label_frame_model, perception_frame_model, it->first,
                      png_path, version);
      size_t label_obstacles_size = label_frame_model.Size();
      labeled_obs_total += label_obstacles_size;
      perception_obs_total += perception_frame_model.Size();
      std::vector<double> velocity_sim_this_frame;
      std::vector<double> velocity_diff_norm_this_frame;
      int match_count =
          ComparePerFrame(it->first, label_frame_model, perception_frame_model,
                          report_model.get(), &velocity_sim_this_frame,
                          &velocity_diff_norm_this_frame);
      std::copy(velocity_sim_this_frame.begin(), velocity_sim_this_frame.end(),
                std::back_inserter(velocity_sim_all_obstacles));
      std::copy(velocity_diff_norm_this_frame.begin(),
                velocity_diff_norm_this_frame.end(),
                std::back_inserter(velocity_diff_norm_whole));
      matches_total += match_count;
      if (perception_frame_model.Size() == 0) {
        zero_ob_frame_count++;
      }
      SaveFrameMatchRes(it->first, match_count, perception_frame_model.Size(),
                        label_obstacles_size, report_model.get());
      std::sort(velocity_sim_this_frame.begin(), velocity_sim_this_frame.end());
      double p50 = GetPn(velocity_sim_this_frame, 50);
      double p90 = GetPn(velocity_sim_this_frame, 95);

      ADEBUG << "frame index " << it->first;
      ADEBUG << "velocity sims: " << GetVecInfo(velocity_sim_this_frame);

      report_model->SetVelocityP50ForFrame(it->first, p50);
      report_model->SetVelocityP95ForFrame(it->first, p90);
    } else {
      AWARN << "ObstacleMatchBase::Match,there is no labeled frame = "
            << it->first;
    }
  }
  double recall_average =
      matches_total / static_cast<double>(labeled_obs_total);
  double precision_average =
      matches_total / static_cast<double>(perception_obs_total);
  std::sort(velocity_sim_all_obstacles.begin(),
            velocity_sim_all_obstacles.end());
  double p50_global = GetPn(velocity_sim_all_obstacles, 50);
  double p95_global = GetPn(velocity_sim_all_obstacles, 95);
  double diff_norm = std::accumulate(velocity_diff_norm_whole.begin(),
                                     velocity_diff_norm_whole.end(), 0.0f);
  diff_norm = velocity_diff_norm_whole.size() == 0
                  ? 0
                  : diff_norm / velocity_diff_norm_whole.size();
  ADEBUG << "all velocity similarities: "
         << GetVecInfo(velocity_sim_all_obstacles);
  report_model->SetRecallAverage(recall_average);
  report_model->SetPrecisionAverage(precision_average);
  report_model->SetVelocityP50Global(p50_global);
  report_model->SetVelocityP95Global(p95_global);
  report_model->SetVelocityDiffNormAverage(diff_norm);
  report_model->SetTotalMatchFrames(frames);
  report_model->SetTotalPerceptionFrames(perception_obstacles_.size());
  report_model->SetTotalLabelFrames(labeled_obstacles_.size());
  report_model->SetZeroObstacleFrames(zero_ob_frame_count);
  AINFO << " match total = " << matches_total << std::endl
        << " recall_average = " << recall_average << std::endl
        << " precision_average = " << precision_average << std::endl
        << " velocity_sim_p50_global = " << p50_global << std::endl
        << " velocity_sim_p95_global = " << p95_global << std::endl
        << " velocity_diff_norm = " << diff_norm << std::endl
        << " labeled_obs_total = " << labeled_obs_total << std::endl
        << " perception_obs_total = " << perception_obs_total << std::endl
        << " match frames = " << frames << std::endl
        << " labeled frames = " << labeled_obstacles_.size() << std::endl
        << " perception frames = " << perception_obstacles_.size();

  return report_model;
}

void ObstacleMatchBase::FilterLabelObstacle(
    const std::map<int, LabelFrameModel> &labeled_obstacles) {
  // ADEBUG << std::setw(4) << localization << std::endl;
  AINFO << "Before filter label_data_ size = " << labeled_obstacles.size();
  if (labeled_obstacles.size() == 0) {
    AERROR << "label_data_ size = 0.FilterLabelObstacle failed";
    return;
  }
  double forward_limit_distance =
      atof(config_->GetValueViaKey("forward_limit_distance").c_str());
  double back_limit_distance =
      atof(config_->GetValueViaKey("back_limit_distance").c_str());
  std::string version = config_->GetValueViaKey("perception");
  int total_obstacles_left = 0;
  int out_of_range = 0;
  int not_on_map = 0;
  for (auto it = labeled_obstacles.begin(); it != labeled_obstacles.end();
       ++it) {
    LocationModel *location_model = GetLocationModelAt(it->first);
    Localization locate;
    if (!location_model) {
      labeled_obstacles_[it->first] = it->second;
      total_obstacles_left += it->second.Size();
    } else {
      location_model->ToLocalizationMsg(&locate);
      LabelFrameModel fram_model;
      labeled_obstacles_[it->first] = fram_model;
      LabelFrameModel label_frame_model_source = it->second;
      size_t size = label_frame_model_source.Size();
      int obs = 0;
      auto timestamp = label_frame_model_source.GetTimeStamp();
      labeled_obstacles_[it->first].SetTimeStamp(timestamp);
      for (size_t i = 0; i < size; ++i) {
        LabeledObstacleModel *ego_front_model =
            label_frame_model_source.GetEgoFrontTypeModelAt(i);
        LabeledObstacleModel *velodyne_model =
            label_frame_model_source.GetVelodyneTypeModelAt(i);
        if (Tool::IsOutOfRange(ego_front_model->GetX(), forward_limit_distance,
                               back_limit_distance)) {
          ++out_of_range;
          continue;
        }
        bool is_on_map = tool_.IsEgoFrontPtOnMap(
            ego_front_model->GetX(), ego_front_model->GetY(),
            ego_front_model->GetZ(), locate, version);
        if (!is_on_map) {
          ++not_on_map;
          continue;
        }
        obs++;
        total_obstacles_left++;
        labeled_obstacles_[it->first].AddObstacleOfVelodyneTypeModel(
            *velodyne_model);
        labeled_obstacles_[it->first].AddObstacleOfEgoFrontTypeModel(
            *ego_front_model);
      }
    }
  }
  AINFO << "After filter label_data_ size = " << labeled_obstacles_.size()
        << " total_obstacles_left = " << total_obstacles_left
        << " not_on_map = " << not_on_map << " out_of_range = " << out_of_range;
}

void ObstacleMatchBase::FilterPerceptionObstacle(
    const std::map<int, PerceptionFrameModel> &perception_obstacles) {
  AINFO << "Before filter perception_data size = "
        << perception_obstacles.size() << std::endl;
  if (perception_obstacles.size() == 0) {
    AERROR << "Error. perception_data size = 0.";
    return;
  }
  double forward_limit_distance =
      atof(config_->GetValueViaKey("forward_limit_distance").c_str());
  double back_limit_distance =
      atof(config_->GetValueViaKey("back_limit_distance").c_str());
  bool use_camera = config_->GetValueViaKey("use_camera") == "true";
  std::string version = config_->GetValueViaKey("perception");
  AINFO << "back_limit_distance = " << back_limit_distance
        << " forward_limit_distance = " << forward_limit_distance
        << " use_camera = " << use_camera;
  int sensor_source_filter_count = 0;
  int total_obstacles_left = 0;
  int out_of_range = 0;
  int not_on_map = 0;
  int radar_obs = 0;
  for (auto it = perception_obstacles.begin(); it != perception_obstacles.end();
       ++it) {
    PerceptionFrameModel perception_frame_model_source = it->second;
    size_t size = perception_frame_model_source.Size();
    LocationModel *location_model =
        perception_frame_model_source.GetLocationModel();
    if (!location_model) {
      continue;
    }
    Localization locate;
    location_model->ToLocalizationMsg(&locate);
    PerceptionFrameModel fram_model;
    fram_model.AddLocationModel(*location_model);
    perception_obstacles_[it->first] = fram_model;
    int obs = 0;
    for (size_t i = 0; i < size; ++i) {
      PerceptionObstacleModel *ego_front_model =
          perception_frame_model_source.GetEgoFrontTypeModelAt(i);
      PerceptionObstacleModel *utm_model =
          perception_frame_model_source.GetUtmTypeModelAt(i);
      if (version == "1") {
        if (!use_camera && ego_front_model->GetSensorSource() ==
                               roadstar::perception::Obstacle::CAMERA) {
          continue;
        }
        if (ego_front_model->GetSensorSource() ==
            roadstar::perception::Obstacle::CAMERA_RADAR) {
          continue;
        }
      } else if (version == "2") {
        if (ego_front_model->GetSensorSource() !=
            roadstar::perception::Obstacle::VELO64) {
          ++sensor_source_filter_count;
          continue;
        }
      } else {
        assert(false);
        AERROR << " no version " << version << " support.";
      }

      if (Tool::IsOutOfRange(ego_front_model->GetX(), forward_limit_distance,
                             back_limit_distance)) {
        ++out_of_range;
        continue;
      }
      bool is_on_map = tool_.IsUtmPointOnMap(
          utm_model->GetX(), utm_model->GetY(), utm_model->GetZ(), locate);
      if (!is_on_map) {
        not_on_map++;
        continue;
      }
      obs++;
      total_obstacles_left++;
      perception_obstacles_[it->first].AddObstacleOfEgoFrontTypeModel(
          *ego_front_model);
      perception_obstacles_[it->first].AddObstacleOfUtmTypeModel(*utm_model);
    }
  }
  AINFO << "After filter, perception_data size = "
        << perception_obstacles_.size()
        << " sensor_source_filter count =  " << sensor_source_filter_count
        << " out_of_range = " << out_of_range << " not_on_map = " << not_on_map
        << " total_obstacles_left = " << total_obstacles_left
        << " radar obs left = " << radar_obs;
}

LocationModel *ObstacleMatchBase::GetLocationModelAt(int frame) {
  if (perception_obstacles_.find(frame) != perception_obstacles_.end()) {
    return perception_obstacles_[frame].GetLocationModel();
  }
  return nullptr;
}

void ObstacleMatchBase::OutputLostPerceptionFramesInfo() {
  for (const auto &it : labeled_obstacles_) {
    if (perception_obstacles_.count(it.first) == 0) {
      AINFO << "perception frame = " << it.first << " is lost.";
    }
  }
}

void ObstacleMatchBase::SaveFrameMatchRes(
    const int &frame, const int &match_count,
    const size_t &perception_obstacles_size, const size_t &label_obstacles_size,
    ReportModel *report_model) {
  if (label_obstacles_size > 0) {
    double percent = match_count / static_cast<double>(label_obstacles_size);
    report_model->SetRecallPercentForFrame(frame, percent);
  } else {
    report_model->SetRecallPercentForFrame(frame, 0);
  }
  if (perception_obstacles_size > 0) {
    double percent =
        match_count / static_cast<double>(perception_obstacles_size);
    report_model->SetPrecisionPercentForFrame(frame, percent);
  } else {
    report_model->SetPrecisionPercentForFrame(frame, 0);
  }

  report_model->SetMatchSizeForFrame(frame, match_count);
  report_model->SetTotalObstaclesOfPerceptionForFrame(
      frame, perception_obstacles_size);
  report_model->SetTotalObstaclesOfLabelForFrame(frame, label_obstacles_size);
}

}  // namespace integration_test
}  // namespace roadstar
