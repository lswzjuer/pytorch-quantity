/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/common/util/util.h"

#include <cctype>
#include <list>
#include <vector>

#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace util {

TEST(Util, unstableremoveif) {
  // vector test.
  {
    std::vector<int> vec{8, 2, 3, 7, 4, 6, 1};
    auto last =
        unstable_remove_if(vec.begin(), vec.end(), [](int x) { return x < 4; });
    auto it = vec.begin();
    EXPECT_EQ(*it++, 8);
    EXPECT_EQ(*it++, 6);
    EXPECT_EQ(*it++, 4);
    EXPECT_EQ(*it++, 7);
    EXPECT_EQ(it, last);
  }
  // list test.
  {
    std::list<int> list{8, 2, 3, 7, 4, 6, 1};
    auto last = unstable_remove_if(list.begin(), list.end(),
                                   [](int x) { return x < 4; });
    auto it = list.begin();
    EXPECT_EQ(*it++, 8);
    EXPECT_EQ(*it++, 6);
    EXPECT_EQ(*it++, 4);
    EXPECT_EQ(*it++, 7);
    EXPECT_EQ(it, last);
  }
  // array test.
  {
    int arr[] = {8, 2, 3, 7, 4, 6, 1};
    auto last = unstable_remove_if(arr, arr + 7, [](int x) { return x < 4; });
    auto it = arr;
    EXPECT_EQ(*it++, 8);
    EXPECT_EQ(*it++, 6);
    EXPECT_EQ(*it++, 4);
    EXPECT_EQ(*it++, 7);
    EXPECT_EQ(it, last);
  }
}

TEST(Util, MaxElement) {
  EXPECT_EQ(3, MaxElement(std::vector<int>{1, 2, 3}));
  EXPECT_FLOAT_EQ(3.3, MaxElement(std::vector<float>{1.1, 2.2, 3.3}));
}

TEST(Util, MinElement) {
  EXPECT_EQ(1, MinElement(std::vector<int>{1, 2, 3}));
  EXPECT_FLOAT_EQ(1.1, MinElement(std::vector<float>{1.1, 2.2, 3.3}));
}

TEST(Util, RemoveIf) {
  std::string str = "Text\n with\tsome \t  spaces\n\n";
  str.erase(std::remove_if(str.begin(), str.end(),
                           [](unsigned char x) { return std::isspace(x); }),
            str.end());
  EXPECT_EQ(str, "Textwithsomespaces");
}

TEST(Util, GetCommandOutputTest) {
  std::string result;
  bool ok = GetCommandOutput("ls / -d", &result);
  ASSERT_TRUE(ok);
  EXPECT_EQ(result, "/\n");
}

struct TestPoint {
  int x_value;
  int y_value;
  int x() const {
    return x_value;
  }
  int y() const {
    return y_value;
  }
};

TEST(Util, PointInPolygonTest) {
  std::vector<TestPoint> case1{{0, 0}, {2, 0}, {1, 2}};
  ASSERT_TRUE(PointInPolygon(1, 1, case1.begin(), case1.end()));
  ASSERT_TRUE(!PointInPolygon(0, 1, case1.begin(), case1.end()));
  std::vector<TestPoint> case2{{0, 0}, {1, 4}, {2, 1}, {3, 3}, {3, 0}};
  ASSERT_TRUE(!PointInPolygon(0, 2, case2.begin(), case2.end()));
  ASSERT_TRUE(!PointInPolygon(2, 2, case2.begin(), case2.end()));
  ASSERT_TRUE(PointInPolygon(1, 1, case2.begin(), case2.end()));
  ASSERT_TRUE(PointInPolygon(1, 2, case2.begin(), case2.end()));
  ASSERT_TRUE(PointInPolygon(1, 3, case2.begin(), case2.end()));
}

TEST(Util, DiscreteFrechetDistanceTest) {
  std::vector<TestPoint> p1{{0, 0}, {2, 0}, {1, 2}};
  std::vector<TestPoint> q1{{0, 0}, {2, 0}, {1, 2}};
  EXPECT_FLOAT_EQ(
      DiscreteFrechetDistance(p1.begin(), p1.end(), q1.begin(), q1.end()), 0.0);
  std::vector<TestPoint> p2{{0, 2}, {-1, 0}, {0, -2}};
  std::vector<TestPoint> q2{{0, 2}, {0, 0}, {0, -2}};
  EXPECT_FLOAT_EQ(
      DiscreteFrechetDistance(p2.begin(), p2.end(), q2.begin(), q2.end()), 1.0);
}

TEST(Util, ProjectPointOnSegmentTest) {
  auto [x, y] = ProjectPointOnSegment(TestPoint{-1, 0}, TestPoint{0, 1},
                                      TestPoint{0, -1});
  EXPECT_FLOAT_EQ(x, 0.0);
  EXPECT_FLOAT_EQ(y, 0.0);

  std::tie(x, y) = ProjectPointOnSegment(TestPoint{2, 0}, TestPoint{-1, -1},
                                         TestPoint{2, 2});
  EXPECT_FLOAT_EQ(x, 1.0);
  EXPECT_FLOAT_EQ(y, 1.0);

  std::tie(x, y) = ProjectPointOnSegment(TestPoint{4, 0}, TestPoint{-1, -1},
                                         TestPoint{1, 1});
  EXPECT_TRUE(std::isnan(x));
  EXPECT_TRUE(std::isnan(y));
}

TEST(Util, DistanceToCurveTest) {
  std::vector<TestPoint> curve({{1, -1}, {1, 1}, {0, 3}});
  EXPECT_FLOAT_EQ(DistanceToCurve(TestPoint{0, 0}, curve.begin(), curve.end()),
                  1.0);
}

TEST(Util, MidPoint2DTest) {
  auto [x, y] = MidPoint2D(TestPoint{0, 0}, TestPoint{1, 1});
  EXPECT_FLOAT_EQ(x, 0.5);
  EXPECT_FLOAT_EQ(y, 0.5);
}

TEST(Util, TestTupleOutput) {
  std::tuple<int, char, std::string> test1{1, 'a', "hello"};
  std::ostringstream os;
  os << test1;
  EXPECT_EQ(os.str(), "(1, a, hello)");
  std::tuple<> test2;
  os.str("");
  os << test2;
  EXPECT_EQ(os.str(), "()");
}

TEST(Util, UniformSlice) {
  std::vector<double> result;
  UniformSlice(0.0, 10.0, 3, &result);
  ASSERT_EQ(4, result.size());
  EXPECT_DOUBLE_EQ(0.0, result[0]);
  EXPECT_DOUBLE_EQ(3.3333333333333335, result[1]);
  EXPECT_DOUBLE_EQ(6.666666666666667, result[2]);
  EXPECT_DOUBLE_EQ(10.0, result[3]);

  UniformSlice(0.0, -5.0, 3, &result);
  ASSERT_EQ(4, result.size());
  EXPECT_DOUBLE_EQ(0.0, result[0]);
  EXPECT_DOUBLE_EQ(-1.666666666666667, result[1]);
  EXPECT_DOUBLE_EQ(-3.3333333333333335, result[2]);
  EXPECT_DOUBLE_EQ(-5.0, result[3]);
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
