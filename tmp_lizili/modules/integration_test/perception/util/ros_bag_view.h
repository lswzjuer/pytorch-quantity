#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_UTIL_ROS_BAG_VIEW_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_UTIL_ROS_BAG_VIEW_H

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <vector>
#include <string>

namespace roadstar {
namespace integration_test {

class RosBagView {
 public:
  explicit RosBagView(const std::string& in_bag);
  ~RosBagView();
  template <class D>
  std::vector<D> GetMessage(const std::string& topic) {
    std::vector<D> ret;
    for (rosbag::MessageInstance const m : rosbag::View(bag_)) {
      std::string str = m.getTopic();
      if (topic == str) {
        ret.emplace_back(*m.instantiate<D>());
      }
    }
    return ret;
  }

 private:
  rosbag::Bag bag_;
};

}  // namespace integration_test
}  // namespace roadstar
#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_UTIL_ROS_BAG_VIEW_H
