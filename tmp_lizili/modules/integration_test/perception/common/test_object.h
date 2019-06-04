#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_COMMON_TEST_OBJECT_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_COMMON_TEST_OBJECT_H

namespace roadstar {
namespace integration_test {

class TestObject {
 public:
  enum test_mode {
    kTestObstacle = 0x1,
    kTestTrafficLight = 0x1 << 1,
  };

  inline static bool
  IsTestObstacleMode(const int& mode) {
    return mode & kTestObstacle;
  }
  inline static bool IsTestTrafficLightMode(const int& mode) {
    return mode & kTestTrafficLight;
  }
};

}  // namespace integration_test
}  // namespace roadstar

#endif
