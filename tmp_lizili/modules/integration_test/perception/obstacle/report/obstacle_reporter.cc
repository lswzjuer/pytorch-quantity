#include "modules/integration_test/perception/obstacle/report/obstacle_reporter.h"

#include <map>

#include "modules/common/log.h"
#include "modules/common/util/file.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "modules/integration_test/perception/obstacle/model/report_model.h"
#include "modules/integration_test/perception/obstacle/serialize/obstacle_dump.h"
#include "modules/integration_test/perception/obstacle/match/match_by_obstacle_center.h"
#include "modules/integration_test/perception/obstacle/match/match_by_obstacle_overlap.h"

namespace roadstar {
namespace integration_test {


void ObstacleReporter::GenerateReport() {
  ObstacleDump dumper(label_json_pathes_, perception_json_path_);
  dumper.Dump();
  std::map<int, LabelFrameModel>& label_obstacles = dumper.GetLabelObstacles();
  std::map<int, PerceptionFrameModel>& perception_obstacles =
      dumper.GetPerceptionObstacles();
  AINFO << "GenerateReport label_obstacles size =" << label_obstacles.size()
        << " perception_obstacles size = " << perception_obstacles.size();
  std::unique_ptr<ObstacleMatchInterface> matcher(
      new MatchByObstacleOverlap(configs_));
  matcher->ComputeLabeledObstacleVelocity(label_obstacles,
                                          perception_obstacles);
  std::shared_ptr<ReportModel> report_model =
      matcher->Match(label_obstacles, perception_obstacles);

  std::string reporter_path = configs_->GetValueViaKey("report_path");
  roadstar::common::util::EnsureDirectory(reporter_path);
  std::string reporter_name = configs_->GetValueViaKey("report_name");
  std::string file = reporter_path + "/" + reporter_name;
  if (report_model) {
    AINFO << "GenerateReport begin "
          << "reporter_path = " << reporter_path
          << " reporter_name = " << reporter_name;
    report_model->SerializeToFile(file);
  }
}

}  // namespace integration_test
}  // namespace roadstar
