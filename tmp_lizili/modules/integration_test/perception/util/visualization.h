#ifndef MODULES_INTEGRATION_PERCEPTION_TEST_UTIL_VISUALIZATION_H
#define MODULES_INTEGRATION_PERCEPTION_TEST_UTIL_VISUALIZATION_H

#include <unsupported/Eigen/CXX11/Tensor>

#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "modules/common/log.h"
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"

namespace roadstar {
namespace integration_test {

class Visualization {
 public:
  Visualization() : vehicle_pos_(500, 1100) {}

  void DrawFramePng(const LabelFrameModel& label_frame,
                    const PerceptionFrameModel& perception_frame,
                    const int64_t frame, const std::string& result_dir,
                    const std::string& version);

 private:
  cv::Point2f GetOffSetPos(const cv::Point2f& source);
  void DrawPolygon(const cv::Mat& mat, const std::vector<cv::Point2f>& polygon,
                   const cv::Scalar& color);
  void DrawText(const cv::Mat& mat, const std::string& text, cv::Point origin,
                cv::Scalar color,  // line color (RGB)
                int fontFace = cv::FONT_HERSHEY_PLAIN,
                double fontScale =
                    2,  // text font scale, text get bigger as this value grows
                int thickness = 2,  // line width
                int lineType = 8,   // line type (4 neighbourhood or 8
                                    // neighbourhood, 8 is the default value)
                bool bottomLeftOrigin = false);  // true='origin at lower left'

  void DrawArrow(const cv::Mat& mat, cv::Point pStart,
                 cv::Scalar color,     // line color
                 double length,        // length of the main line in the arrow
                 double short_length,  // length of the head line in the arrow
                 double heading,       // the direction of the arrow
                 double alpha,  // the angle between main line and head line
                 int thickness = 2,  // line width
                 int lineType = 8);  // line type (4 neighbourhood or 8
                                     // neighbourhood, 8 is the default value)
  enum PngSize {
    kPngSizeX = 1500,
    kPngSizeY = 1000,
  };

  cv::Point2f vehicle_pos_;
};

}  // namespace integration_test
}  // namespace roadstar
#endif  // MODULES_INTEGRATION_PERCEPTION_TEST_UTIL_VISUALIZATION_H
