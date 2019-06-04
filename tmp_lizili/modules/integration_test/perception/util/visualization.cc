#include "modules/integration_test/perception/util/visualization.h"

#include <sstream>

namespace roadstar {
namespace integration_test {
using PointENU = roadstar::common::PointENU;

void Visualization::DrawFramePng(const LabelFrameModel& label_frame,
                                 const PerceptionFrameModel& perception_frame,
                                 const int64_t frame,
                                 const std::string& result_dir,
                                 const std::string& version) {
  // Visualization: Init
  cv::Mat mat = cv::Mat::zeros(kPngSizeX, kPngSizeY, CV_8UC3);
  // draw vehicle self
  cv::Scalar white = cv::Scalar(255, 255, 255);
  cv::Scalar green = cv::Scalar(0, 255, 0);
  cv::circle(mat, vehicle_pos_, 10, white, -1);
  int line_y = static_cast<int>(kPngSizeX - 400 - 70 / 0.1);
  cv::line(mat, cv::Point(static_cast<int>(0), line_y),
           cv::Point(static_cast<int>(kPngSizeY), line_y), white, 2);
  // Visualization: Drawing bounding box
  cv::Scalar perception_color = green;
  DrawText(mat, "green box : perception", cv::Point(10, line_y + 30), green);
  std::size_t perception_size = perception_frame.Size();
  PerceptionFrameModel* perception_frame_model =
      const_cast<PerceptionFrameModel*>(&perception_frame);
  LocationModel* location_model = perception_frame_model->GetLocationModel();
  for (std::size_t index = 0; index < perception_size; ++index) {
    PerceptionObstacleModel* obstacle_model =
        perception_frame_model->GetUtmTypeModelAt(index);
    std::vector<PointENU> org_polygon =
        obstacle_model->GetEgoFrontTypePolygon(*location_model, version);
    std::vector<cv::Point2f> dest_polygon;
    for (const auto& pt : org_polygon) {
      cv::Point2f pt2f(pt.x() / 0.1, pt.y() / 0.1);
      dest_polygon.emplace_back(GetOffSetPos(pt2f));
    }
    DrawPolygon(mat, dest_polygon, perception_color);
    cv::Point center;
    center.x = (dest_polygon[0].x + dest_polygon[1].x + dest_polygon[2].x +
                dest_polygon[3].x) /
               4;
    center.y = (dest_polygon[0].y + dest_polygon[1].y + dest_polygon[2].y +
                dest_polygon[3].y) /
               4;
    DrawArrow(mat, center, perception_color,
              obstacle_model->GetVelocity() / 0.2, 20,
              obstacle_model->GetHeading(), 30);
  }
  cv::Scalar label_color = white;
  DrawText(mat, "white box : label", cv::Point(10, line_y + 60), white);
  std::size_t label_size = label_frame.Size();
  LabelFrameModel* label_frame_model =
      const_cast<LabelFrameModel*>(&label_frame);
  for (std::size_t index = 0; index < label_size; ++index) {
    LabeledObstacleModel* obstacle_model =
        label_frame_model->GetVelodyneTypeModelAt(index);
    std::vector<PointENU> org_polygon =
        obstacle_model->GetEgoFrontTypePolygon(version);
    std::vector<cv::Point2f> dest_polygon;
    for (const auto& pt : org_polygon) {
      cv::Point2f pt2f(pt.x() / 0.1, pt.y() / 0.1);
      dest_polygon.emplace_back(GetOffSetPos(pt2f));
    }
    DrawPolygon(mat, dest_polygon, label_color);
    cv::Point center;
    center.x = (dest_polygon[0].x + dest_polygon[1].x + dest_polygon[2].x +
                dest_polygon[3].x) /
               4;
    center.y = (dest_polygon[0].y + dest_polygon[1].y + dest_polygon[2].y +
                dest_polygon[3].y) /
               4;
    DrawArrow(mat, center, label_color, obstacle_model->GetVelocity() / 0.2, 20,
              obstacle_model->GetHeading(), 30);
    cv::Point id_pos;
    id_pos.y = center.y;
    id_pos.x = center.x + 20;

    std::stringstream ss;
    ss << "id: " << obstacle_model->GetId();
    DrawText(mat, ss.str(), id_pos, label_color);
  }
  cv::imwrite(result_dir + "/" + std::to_string(frame) + ".png", mat);
}

cv::Point2f Visualization::GetOffSetPos(const cv::Point2f& source) {
  return cv::Point2f(500 - source.y, kPngSizeX - 400 - source.x);
}

void Visualization::DrawPolygon(const cv::Mat& mat,
                                const std::vector<cv::Point2f>& polygon,
                                const cv::Scalar& color) {
  std::size_t size = polygon.size();
  for (std::size_t index = 0; index < size; ++index) {
    cv::line(mat, polygon[index], polygon[(index + 1) % size], color, 2);
  }
}

void Visualization::DrawText(
    const cv::Mat& mat, const std::string& text, cv::Point origin,
    cv::Scalar color,  // 线条的颜色（RGB）
    int fontFace,
    double fontScale,      // 尺寸因子，值越大文字越大
    int thickness,         // 线条宽度
    int lineType,          // 线型（4邻域或8邻域，默认8邻域）
    bool bottomLeftOrigin  // true='origin at lower left'
    ) {
  cv::putText(mat, text, origin, fontFace, fontScale, color, thickness,
              lineType, bottomLeftOrigin);
}

void Visualization::DrawArrow(const cv::Mat& mat, cv::Point pStart, cv::Scalar color,
                              double length, double short_length,
                              double heading, double alpha, int thickness,
                              int lineType) {
  const double PI = 3.1415926;

  cv::Point pEnd;
  pEnd.x = pStart.x + length * cos(heading);
  pEnd.y = pStart.y - length * sin(heading);
  cv::line(mat, pStart, pEnd, color, thickness, lineType);

  cv::Point arrow;
  double angle = atan2(static_cast<double>(pStart.y - pEnd.y),
                       static_cast<double>(pStart.x - pEnd.x));

  arrow.x = pEnd.x + short_length * cos(angle + PI * alpha / 180);
  arrow.y = pEnd.y + short_length * sin(angle + PI * alpha / 180);
  cv::line(mat, pEnd, arrow, color, thickness, lineType);

  arrow.x = pEnd.x + short_length * cos(angle - PI * alpha / 180);
  arrow.y = pEnd.y + short_length * sin(angle - PI * alpha / 180);
  cv::line(mat, pEnd, arrow, color, thickness, lineType);
}

}  // namespace integration_test
}  // namespace roadstar
