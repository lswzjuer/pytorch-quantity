#include <chrono>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "modules/common/util/thread_pool.h"

namespace roadstar {
namespace common {
namespace util {

class foo {
 public:
  explicit foo(int x) {
    x_ = x;
  }
  int bar(int x) {
    return x * x + x_;
  }

 private:
  int x_;
};

TEST(ThreadPool, General) {
  ThreadPool pool(4);
  std::vector<std::future<int>> results;

  // lamdal function test
  for (int i = 0; i < 8; ++i) {
    results.emplace_back(pool.Enqueue([i] {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      return i * i;
    }));
  }
  for (size_t i = 0; i < results.size(); i++) {
    EXPECT_EQ(results[i].get(), i * i);
  }

  // class function test
  foo f(5);
  // (function pointer, this pointer, argument)
  auto result = pool.Enqueue(&foo::bar, &f, 4);
  EXPECT_EQ(result.get(), 21);
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
