#ifndef MODULES_COMMON_HDMAP_CLIENT_MAP_CLIENT_H
#define MODULES_COMMON_HDMAP_CLIENT_MAP_CLIENT_H

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "grpc++/grpc++.h"

#include "modules/common/status/status.h"
#include "modules/msgs/hdmap/proto/hdmap_common.pb.h"
#include "modules/msgs/hdmap/proto/lanemarker.pb.h"
#include "modules/msgs/hdmap/proto/rpc_service.grpc.pb.h"

namespace roadstar {
namespace common {
using Lanemarkers = std::vector<roadstar::hdmap::Lanemarker>;

class MapClient {
 public:
  using Id = unsigned int;

  explicit MapClient(std::unique_ptr<roadstar::hdmap::RpcServiceNode::Stub>);

  MapClient();

  ~MapClient() {}

  Status GetAvailableMaps(
      std::unordered_map<std::string, roadstar::hdmap::MapConfig>*
          available_maps,
      const int map_client_timeout_ms = -1);

  Status GetLocalLanemarkers(const double x, const double y,
                             const double forward_distance,
                             const double backward_distance,
                             std::vector<Lanemarkers>* lanemarkers,
                             const int map_client_timeout_ms = -1);

  Status GetLocalBoundary(const double x, const double y, const double radius,
                          std::vector<Polygon>* polygons,
                          const int map_client_timeout_ms = -1);

  Status SetMap(const std::string& map_name,
                const int map_client_timeout_ms = -1);

  Status GetCurrMap(std::string* const curr_map_name,
                    const int map_client_timeout_ms = -1);

  Status SetRoutingPoints(const std::vector<PointENU>& routing_points,
                          const int map_client_timeout_ms = -1);

  Status SetRoute(const std::string& route_name,
                  const int map_client_timeout_ms = -1);

  Status GetRoutingPath(roadstar::hdmap::Path* routing_path,
                        const int map_client_timeout_ms = -1);

  Status GetCurrRoute(std::string* const curr_route_name,
                      const int map_client_timeout_ms = -1);

  Status GetLocalMap(const double x, const double y, const double radius,
                     roadstar::hdmap::MapProto* local_map,
                     const int map_client_timeout_ms = -1) const;

  Status GetLocalPath(const double x, const double y,
                      const double forward_distance,
                      const double backward_distance,
                      roadstar::hdmap::Path* path,
                      const int map_client_timeout_ms = -1);

  Status GetPointOnRoad(
      const std::vector<PointENU>& points, const PointENU& localization,
      std::vector<roadstar::hdmap::PointInfo>* const point_infos,
      double radius = 200, const int map_client_timeout_ms = -1);

  Status RetrieveMapElements(
      const std::vector<roadstar::hdmap::MapUnit>& map_units,
      roadstar::hdmap::MapElements* map_elements,
      const int map_client_timeout_ms = -1);

 private:
  /**
   * @brief Return a list of map elements according to the path unit ids.
   * @param elementIds The
   */
  Status RetrieveMapElements(const google::protobuf::RepeatedPtrField<
                                 roadstar::hdmap::MapUnit>& map_units,
                             const int map_client_timeout_ms);

 private:
  // map rpc service stub
  std::unique_ptr<roadstar::hdmap::RpcServiceNode::Stub> stub_ = nullptr;

  // cached data mutex.
  mutable std::mutex mutex_;

  // Cached map elements ids.
  std::list<roadstar::hdmap::MapUnit> cached_map_units_;

  // Cached map elemnets.
  std::unordered_map<Id, roadstar::hdmap::Section> sections_;

  // Cached map connections.
  std::unordered_map<Id, roadstar::hdmap::Connection> connections_;
};

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_HDMAP_CLIENT_MAP_CLIENT_H
