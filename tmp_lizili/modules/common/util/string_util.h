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

/**
 * @file
 * @brief Some string util functions.
 */

#ifndef MODULES_COMMON_UTIL_STRING_UTIL_H_
#define MODULES_COMMON_UTIL_STRING_UTIL_H_

#include <google/protobuf/stubs/stringprintf.h>
#include <google/protobuf/stubs/strutil.h>
#include <algorithm>
#include <sstream>
#include <string>

/**
 * @namespace roadstar::common::util
 * @brief roadstar::common::util
 */
namespace roadstar {
namespace common {
namespace util {

// Expose some useful utils from protobuf.
using google::protobuf::Join;
using google::protobuf::StrAppend;
using google::protobuf::StrCat;
using google::protobuf::StringPiece;
using google::protobuf::StringPrintf;

/**
 * @brief Check if a string ends with a pattern.
 * @param ori The original string. To see if it ends with a specified pattern.
 * @param pat The target pattern. To see if the original string ends with it.
 * @return Whether the original string ends with the specified pattern.
 */
inline bool EndWith(const std::string &ori, const std::string &pat) {
  return StringPiece(ori).ends_with(pat);
}

/**
 * @brief Concat parameters to a string, e.g.: StrCat("age = ", 32)
 * @return String of concated parameters.
 */
template <typename... T>
std::string StrCat(const T &... args) {
  std::ostringstream oss;
  (oss << ... << args);
  return oss.str();
}

template <typename T>
std::string Print(const T &val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

/**
 * @brief Make arrays, conatiners and iterators printable.
 *
 * Usage:
 *   vector<int> vec = {1, 2, 3};
 *   std::cout << PrintIter(vec);
 *   std::cout << PrintIter(vec, ",");
 *   std::cout << PrintIter(vec.begin(), vec.end());
 *   std::cout << PrintIter(vec.begin(), vec.end(), "|");
 *
 *   int array[] = {1, 2, 3};
 *   std::cout << PrintIter(array);
 *   std::cout << PrintIter(array, "|");
 *   std::cout << PrintIter(array + 0, array + 10, "|");
 */
template <typename Iter>
std::string PrintIter(const Iter &begin, const Iter &end,
                      const std::string &delimiter = " ") {
  std::string result;
  Join(begin, end, delimiter.c_str(), &result);
  return result;
}

template <typename Container>
std::string PrintIter(const Container &container,
                      const std::string &delimiter = " ") {
  return PrintIter(container.begin(), container.end(), delimiter);
}

template <typename T, int Length>
std::string PrintIter(T (&array)[Length], T *end,
                      const std::string &delimiter = " ") {
  std::string result;
  Join(array, end, delimiter.c_str(), &result);
  return result;
}

template <typename T, int Length>
std::string PrintIter(T (&array)[Length], const std::string &delimiter = " ") {
  return PrintIter(array, array + Length, delimiter);
}

/**
 * @brief Make conatiners and iterators printable. Similar to PrintIter but
 *        output the DebugString().
 */
template <typename Iter>
std::string PrintDebugStringIter(const Iter &begin, const Iter &end,
                                 const std::string &delimiter = " ") {
  std::string result;
  for (auto iter = begin; iter != end; ++iter) {
    if (iter == begin) {
      StrAppend(&result, iter->DebugString());
    } else {
      StrAppend(&result, delimiter, iter->DebugString());
    }
  }
  return result;
}

template <typename Container>
std::string PrintDebugStringIter(const Container &container,
                                 const std::string &delimiter = " ") {
  return PrintDebugStringIter(container.begin(), container.end(), delimiter);
}

inline std::string Trim(const std::string &source) {
  int l, r;
  for (l = 0; l < source.size() && std::isspace(source[l]); l++) {
  }
  if (l >= source.size()) return "";
  for (r = static_cast<int>(source.size() - 1);
       r >= 0 && std::isspace(source[r]); r--) {
  }
  return source.substr(l, r - l + 1);
}

/**
 * @brief Convert 'ABC_DEF' like string to 'AbcDef' format
 */
inline std::string ToPascalName(const std::string &name) {
  std::string result;
  result.reserve(name.size());
  bool flag = true;  // indicate whether the first letter
  for (const auto &ch : name) {
    if (ch == '_') {
      flag = true;
      continue;
    }

    if (!std::isalpha(ch)) {
      continue;
    }

    if (flag == true) {
      result.push_back(std::toupper(ch));
      flag = false;
    } else {
      result.push_back(std::tolower(ch));
    }
  }
  return result;
}

}  // namespace util
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_UTIL_STRING_UTIL_H_
