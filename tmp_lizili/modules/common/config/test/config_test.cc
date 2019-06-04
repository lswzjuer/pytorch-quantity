#include "modules/common/config/config.h"
#include <gtest/gtest.h>
#include "modules/common/config/config_manager.h"
#include "modules/common/config/test/config_test.pb.h"

namespace roadstar::common {

class ConfigManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConfigManager::Init("modules/common/config/test/configs");
  }
  void TearDown() override {}
};
TEST_F(ConfigManagerTest, Test) {
  auto config_a = Config<common::ConfigTest>::Get("a");
  auto config_b = Config<common::ConfigTest2>::Get("test/b");

  EXPECT_EQ(config_a.header().id(), 1);
  ASSERT_TRUE(config_a.has_name());
  EXPECT_EQ(config_a.name(), "a");

  ASSERT_TRUE(config_b.has_test());
  EXPECT_EQ(config_b.test(), "b");
}
}  // namespace roadstar::common
