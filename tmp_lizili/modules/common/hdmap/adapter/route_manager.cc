#include "modules/common/hdmap/adapter/route_manager.h"

#include "modules/common/hdmap/client/map_client.h"
#include "modules/common/log.h"

namespace roadstar {
namespace common {
namespace hdmap {

namespace {
constexpr size_t kMapClientTimeoutMs = 100;
}

RouteManager::RouteManager() {
  SyncRoute();
}

bool RouteManager::ReRoute(
    const std::vector<::roadstar::common::PointENU> &route_request) {
  MapClient client;
  std::vector<::roadstar::hdmap::MapUnit> path;
  auto status = client.Route(route_request, &path, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR << "Failed to route : " << status.error_message();
    return false;
  }

  // sync path.
  std::lock_guard<std::mutex> lck(mutex_);
  path_ = std::move(path);
  map_unit_index_.clear();
  for (std::size_t i = 0; i < path_.size(); ++i) {
    map_unit_index_.emplace(path_[i], i);
  }

  return true;
}

bool RouteManager::SyncRoute() {
  MapClient client;
  std::vector<::roadstar::hdmap::MapUnit> path;
  auto status = client.GetRoute(&path, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR << "Failed to route : " << status.error_message();
    return false;
  }

  // sync path.
  std::lock_guard<std::mutex> lck(mutex_);
  path_ = std::move(path);
  map_unit_index_.clear();
  for (std::size_t i = 0; i < path_.size(); ++i) {
    map_unit_index_.emplace(path_[i], i);
  }

  return true;
}

bool RouteManager::IsOnPath(const ::roadstar::hdmap::MapUnit &map_unit) {
  {
    std::lock_guard<std::mutex> lck(instance()->mutex_);
    if (instance()->map_unit_index_.find(map_unit) !=
        instance()->map_unit_index_.end()) {
      return true;
    }
  }
  instance()->SyncRoute();
  std::lock_guard<std::mutex> lck(instance()->mutex_);
  return instance()->map_unit_index_.find(map_unit) !=
         instance()->map_unit_index_.end();
}

bool RouteManager::GetNextMapUnit(
    const ::roadstar::hdmap::MapUnit &curr_map_unit,
    ::roadstar::hdmap::MapUnit *const next_map_unit) {
  return instance()->InternalGetNextKthMapUnit(curr_map_unit, 1, next_map_unit);
}

bool RouteManager::GetPreviousMapUnit(
    const ::roadstar::hdmap::MapUnit &curr_map_unit,
    ::roadstar ::hdmap::MapUnit *const previous_map_unit) {
  return instance()->InternalGetNextKthMapUnit(curr_map_unit, -1,
                                               previous_map_unit);
}

bool RouteManager::InternalGetNextKthMapUnit(
    const ::roadstar::hdmap::MapUnit &map_unit, int k,
    ::roadstar::hdmap::MapUnit *const next_map_unit) {
  std::lock_guard<std::mutex> lck(mutex_);
  if (path_.empty()) {
    AERROR << "The path is empty!";
    return false;
  }
  auto iter = map_unit_index_.find(map_unit);
  if (iter == map_unit_index_.end()) {
    AERROR << "The " << map_unit.DebugString() << " is not on the path.";
    return false;
  }

  if (iter->second + k >= path_.size() || iter->second + k < 0) {
    return false;
  }

  *next_map_unit = path_[iter->second + k];

  return true;
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
