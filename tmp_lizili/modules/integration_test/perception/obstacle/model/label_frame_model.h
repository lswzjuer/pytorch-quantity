#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_LABEL_FRAME_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_LABEL_FRAME_MODEL_H

#include <vector>
#include <string>
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"

namespace roadstar {
namespace integration_test {

class LabelFrameModel {
 public:
  LabelFrameModel() = default;
  explicit LabelFrameModel(const json& value) {
    FromJson(value);
  }

  void AddObstacleOfVelodyneTypeModel(const LabeledObstacleModel& model);
  void AddObstacleOfEgoFrontTypeModel(const LabeledObstacleModel& model);

  LabeledObstacleModel* GetEgoFrontTypeModelAt(size_t i);
  LabeledObstacleModel* GetVelodyneTypeModelAt(size_t i);
  const LabeledObstacleModel* GetEgoFrontTypeModelAt(size_t i) const;
  const LabeledObstacleModel* GetVelodyneTypeModelAt(size_t i) const;

  size_t Size() const;

  json ToJson() const;
  void FromJson(const json& value);

  json RawDataToJson() const;
  void FromRawDataJson(const json& value);

  void SetTimeStamp(double value);
  double GetTimeStamp() const;

  bool ParseFromFile(const std::string& file);
  bool SerializeToFile(const std::string& file) const;

  bool ParseRawDataFromFile(const std::string& file);
  bool SerializeRawDataToFile(const std::string& file) const;

  void Print();

  bool EraseObstacleAt(size_t index);
  bool HasObstacleId(size_t id) const;
  int GetIndexOfObstacleId(size_t id) const;
  LabeledObstacleModel* GetEgoFrontTypeModelByID(std::size_t id);
  LabeledObstacleModel* GetVelodyneTypeModelByID(std::size_t id);

 private:
  std::vector<LabeledObstacleModel> ego_front_modeles_;
  std::vector<LabeledObstacleModel> velodyne_modeles_;
  double time_stamp_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
