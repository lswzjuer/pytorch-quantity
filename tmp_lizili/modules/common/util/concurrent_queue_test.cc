#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "modules/common/util/concurrent_queue.h"

namespace roadstar {
namespace common {
namespace util {

TEST(ConcurrentQueue, General) {
  std::vector<int> t{1, 2, 3};
  ConcurrentQueue<std::vector<int>> concurrent_queue(3);
  concurrent_queue.Push(t);
  EXPECT_EQ(t.size(), 3);
  concurrent_queue.Push(std::move(t));
  EXPECT_EQ(t.size(), 0);
  concurrent_queue.Push(std::in_place, 2, 2);
  auto t1 = concurrent_queue.Pop();
  EXPECT_EQ(t1.size(), 3);
  EXPECT_EQ(t1[0], 1);
  EXPECT_EQ(t1[1], 2);
  EXPECT_EQ(t1[2], 3);
  concurrent_queue.Pop(&t);
  EXPECT_EQ(t.size(), 3);
  EXPECT_EQ(t[0], 1);
  EXPECT_EQ(t[1], 2);
  EXPECT_EQ(t[2], 3);
  t = concurrent_queue.Pop();
  EXPECT_EQ(t.size(), 2);
  EXPECT_EQ(t[0], 2);
  EXPECT_EQ(t[1], 2);
}

TEST(ConcurrentQueue, UniquePtr) {
  std::unique_ptr<int> a = std::make_unique<int>(1);

  ConcurrentQueue<std::unique_ptr<int>> concurrent_queue(3);

  concurrent_queue.Push(std::move(a));
  concurrent_queue.Push(std::in_place, std::make_unique<int>(2));

  EXPECT_EQ(a, nullptr);

  auto t = concurrent_queue.Pop();
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(*t, 1);

  concurrent_queue.Pop(&t);
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(*t, 2);
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
