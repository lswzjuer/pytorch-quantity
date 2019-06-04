#pragma once 

#include "boost/functional/hash.hpp"

#include "modules/msgs/hdmap/proto/hdmap_common.pb.h"

namespace std {

template <>
struct hash<::roadstar::hdmap::MapUnit> {
  std::size_t operator()(const ::roadstar::hdmap::MapUnit &map_unit) const {
    std::size_t seed;
    boost::hash_combine(seed, map_unit.id());
    boost::hash_combine(seed, map_unit.type());

    return seed;
  }
};

template <>
struct equal_to<::roadstar::hdmap::MapUnit> {
  bool operator()(const ::roadstar::hdmap::MapUnit &lhs,
                  const roadstar::hdmap::MapUnit &rhs) const {
    return lhs.id() == rhs.id() && lhs.type() == rhs.type();
  }
};

}  // namespace std
