#pragma once

#include <future>

#include "modules/common/hdmap/common/local_map.h"
#include "modules/common/macro.h"
#include "modules/common/util/thread_pool.h"

namespace roadstar {
namespace common {
namespace hdmap {

/**
 * @class LocalMapManager
 * @brief This class provide a interface to retrieve local map to interact with
 * hdmap server. It also provide a mechanism to asynchronously update the local
 * map for caching the map elements.
 *
 * \note Most of case, you should add the AsyncUpdate to the callback of
 * localization. Thoungh it is not necessary, it will greatly reduce the time
 * cost for GetLocalMap.
 *
 * \note LocalMapManager::Observe() is thread safe, but calling it from multiple
 * threads may introduce unexpected behavior. A common usage is calling Observe
 * only once at the begining of each process cycle and all use a same map.
 *
 * */
class LocalMapManager {
 public:
  /**
   * @brief Get the local map at the given location with the given radius.
   * @param location The requested location (the center of local map).
   * @param radius The radius of local map.
   * @param local_map The requested local map.
   * @param timeout_ms The timeout duration in milliseconds.
   * @return A bool devoting the requested local map is valid. False for failing
   * to requested or timeout, True for valid.
   * */
  static bool GetLocalMap(const ::roadstar::common::PointENU &location,
                          double radius, LocalMap *const local_map,
                          size_t timeout_ms = 0) {
    return instance()->InternalGetLocalMap(location, radius, local_map);
  }

  /**
   * @brief Make a copy of map at a specific location and radius to create a
   * view of local map. And this procedure may timeout and it will return false
   * if timeout happens.
   * @param location The center of requested local map.
   * @param radius The radius of requested local map.
   * @param timeout_ms The timeout milliseconds to request local map.
   * @return A bool value indicates whether the observation is sucessfully.
   * */
  static bool Observe(const ::roadstar::common::PointENU &location,
                      double radius, size_t timeout_ms) {
    std::lock_guard<std::mutex> lck(instance()->observed_map_mutex_);
    return GetLocalMap(location, radius, &instance()->observed_map_,
                       timeout_ms);
  }

  /**
   * @brief Get the latested observad local map.
   * @return The local map.
   * */
  static LocalMap GetLatestObserved() {
    std::lock_guard<std::mutex> lck(instance()->observed_map_mutex_);
    return instance()->observed_map_;
  }

  /**
   * @brief Async update the cached local map.
   * @param location The requested location (the center of local map).
   * @param radius The radius of local map.
   * */
  static void AsyncUpdate(const ::roadstar::common::PointENU &location) {
    instance()->async_update_thread_.Enqueue(
        &LocalMapManager::InternalAsyncUpdate, instance(), location);
  }

 private:
  bool InternalGetLocalMap(const ::roadstar::common::PointENU &location,
                           double radius, LocalMap *const local_map);

  void InternalAsyncUpdate(const ::roadstar::common::PointENU &point);

 private:
  // The cached local map.
  LocalMap local_map_;

  // Observed local map.
  LocalMap observed_map_;

  // The mutex for local map.
  mutable std::mutex mutex_;

  // The mutex for observed map.
  mutable std::mutex observed_map_mutex_;

  // The thread used to update the cached local map.
  ThreadPool async_update_thread_;

  DECLARE_SINGLETON(LocalMapManager);
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
