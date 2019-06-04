#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "modules/common/hdmap/base/map_unit_hash.h"
#include "modules/common/hdmap/client/map_client.h"
#include "modules/common/macro.h"
#include "modules/common/util/thread_pool.h"

namespace roadstar {
namespace common {
namespace hdmap {

class RouteManager {
 public:
  static bool IsOnPath(const ::roadstar::hdmap::MapUnit &map_unit);

  static bool GetNextMapUnit(const ::roadstar::hdmap::MapUnit &map_unit,
                             ::roadstar::hdmap::MapUnit *const next_map_unit);

  static bool GetPreviousMapUnit(
      const ::roadstar::hdmap::MapUnit &map_unit,
      ::roadstar::hdmap::MapUnit *const next_map_unit);

 private:
  bool InternalGetNextKthMapUnit(
      const ::roadstar::hdmap::MapUnit &map_unit, int k,
      ::roadstar::hdmap::MapUnit *const next_map_unit);

  bool ReRoute(const std::vector<::roadstar::common::PointENU> &route_request);

  bool SyncRoute();

 private:
  std::unordered_map<::roadstar::hdmap::MapUnit, std::size_t> map_unit_index_;

  std::vector<::roadstar::hdmap::MapUnit> path_;

  mutable std::mutex mutex_;

  DECLARE_SINGLETON(RouteManager);
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
