#ifndef MODULES_COMMON_SENSOR_SOURCE_H_
#define MODULES_COMMON_SENSOR_SOURCE_H_

#include <algorithm>
#include <unordered_map>
#include <string>

#include "modules/common/proto/sensor_source.pb.h"
#include "modules/common/util/hash.h"

namespace roadstar {
namespace common {
namespace sensor {

/*
#define IS_TYPE(type)                                                         \
inline bool Is##type(const SensorSource &source) {                          \
  return SensorSource_Name(source).compare(0u, sizeof(#type) - 1, #type) == \
         0;                                                                 \
}

// IS_TYPE(Camera)
// IS_TYPE(Radar)
// IS_TYPE(Lidar)
// IS_TYPE(Fusion)

// #undef IS_TYPE

*/

inline std::string Name(const SensorSource &source) {
  const auto &name = SensorSource_Name(source);
  auto iter = std::find_if(++name.begin(), name.end(),
                           [](const char &c) { return c >= 'A' && c <= 'Z'; });
  return std::string(name.begin(), iter) + '(' + std::string(iter, name.end()) +
         ')';
}

inline SensorType GetSensorType(const SensorSource &source) {
  static std::unordered_map<std::string, SensorType> type_map = ([]() {
    std::unordered_map<std::string, SensorType> result;
    for (size_t i = SensorType_MIN; i < SensorType_MAX; ++i) {
      result[SensorType_Name(static_cast<SensorType>(i))] =
          static_cast<SensorType>(i);
    }
    return result;
  })();
  const auto &name = SensorSource_Name(source);
  auto iter = std::find_if(++name.begin(), name.end(),
                           [](const char &c) { return c >= 'A' && c <= 'Z'; });
  return type_map[std::string(name.begin(), iter)];
}

template <SensorType type>
inline bool Is(const SensorSource &source) {
  return GetSensorType(source) == type;
}

inline bool Is(const SensorSource &source, const SensorType &type) {
  return GetSensorType(source) == type;
}

}  // namespace sensor
}  // namespace common
}  // namespace roadstar

ENABLE_ENUM_HASH(::roadstar::common::sensor::SensorSource)
ENABLE_ENUM_HASH(::roadstar::common::sensor::SensorType)

#endif  // MODULES_COMMON_SENSOR_SOURCE_H_
