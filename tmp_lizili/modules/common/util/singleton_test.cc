#include <chrono>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "modules/common/util/singleton.h"

namespace roadstar {
namespace common {
namespace util {

class Foo {
 public:
  explicit Foo() {
    x_ = 10;
  }
  int Bar(int x) {
    return x * x + x_;
  }

 private:
  int x_;
};

class FooSingleton : public Singleton<Foo, FooSingleton> {};

TEST(Singleton, General) {
  EXPECT_EQ(FooSingleton()->Bar(10), 110);
  EXPECT_EQ((*FooSingleton()).Bar(10), 110);
  EXPECT_EQ(FooSingleton::instance(), FooSingleton::instance());
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
