#include "modules/integration_test/perception/util/ros_bag_view.h"

namespace roadstar {
namespace integration_test {

RosBagView::RosBagView(const std::string& in_bag) {
  bag_.open(in_bag);
}

RosBagView::~RosBagView() {
  bag_.close();
}

}  // namespace integration_test
}  // namespace roadstar
