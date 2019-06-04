#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_PERCEPTION_FRAME_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_PERCEPTION_FRAME_MODEL_H

#include <vector>
#include <string>

#include "modules/integration_test/perception/obstacle/model/location_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"

namespace roadstar {
namespace integration_test {

class PerceptionFrameModel {
 public:
  void AddObstacleOfUtmTypeModel(const PerceptionObstacleModel& model);
  void AddObstacleOfEgoFrontTypeModel(const PerceptionObstacleModel& model);
  void AddLocationModel(const LocationModel& model);
  PerceptionObstacleModel* GetEgoFrontTypeModelAt(size_t i);
  const PerceptionObstacleModel* GetEgoFrontTypeModelAt(size_t i) const;
  PerceptionObstacleModel* GetUtmTypeModelAt(size_t i);
  const PerceptionObstacleModel* GetUtmTypeModelAt(size_t i) const;
  LocationModel* GetLocationModel();
  const LocationModel* GetLocationModel() const;
  size_t Size() const;

  json ToJson();
  void FromJson(const json& value);

  bool ParseFromFile(const std::string& file);
  bool SerializeToFile(const std::string& file);
  bool ParseRawDataFromFile(const std::string& file);
  bool SerializeRawDataToFile(const std::string& file);
  void Print();

 private:
  std::vector<PerceptionObstacleModel> ego_front_modeles_;
  std::vector<PerceptionObstacleModel> utm_modeles_;
  LocationModel location_model_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
