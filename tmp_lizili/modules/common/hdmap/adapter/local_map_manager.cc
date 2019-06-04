#include "modules/common/hdmap/adapter/local_map_manager.h"

namespace roadstar {
namespace common {
namespace hdmap {

namespace {

// Used for pre-caching local map and reduce the frequency to call rpc map
// server.
constexpr double kLocalMapUpdateRadius = 500;
constexpr double kLocalMapRequestRadius = 2000;

}  // namespace

LocalMapManager::LocalMapManager() : async_update_thread_(1, 1) {}

bool LocalMapManager::InternalGetLocalMap(
    const ::roadstar::common::PointENU &location, double radius,
    LocalMap *const local_map) {
  {
    std::lock_guard<std::mutex> lck(mutex_);
    *local_map = local_map_;
  }
  double dist = std::hypot(location.x() - local_map->center().x(),
                           location.y() - local_map->center().y());
  if (radius + dist <= local_map->radius()) {
    local_map->Shrink(location, radius);
    return true;
  }
  if (!local_map->Update(location, radius)) {
    return false;
  }

  std::lock_guard<std::mutex> lck(mutex_);
  local_map_ = *local_map;

  return true;
}

void LocalMapManager::InternalAsyncUpdate(
    const ::roadstar::common::PointENU &location) {
  LocalMap new_local_map;
  {
    std::lock_guard<std::mutex> lck(mutex_);
    auto local_map_center = local_map_.center();
    double dist = std::hypot(location.x() - local_map_center.x(),
                             location.y() - local_map_center.y());
    if (dist <= kLocalMapUpdateRadius) {
      return;
    }
    new_local_map = local_map_;
  }

  // Update the local map.
  if (new_local_map.Update(location, kLocalMapRequestRadius)) {
    std::lock_guard<std::mutex> lck(mutex_);
    local_map_ = new_local_map;
  }
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
