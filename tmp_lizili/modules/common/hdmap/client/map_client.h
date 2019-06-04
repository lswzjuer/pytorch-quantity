#pragma once

#include <memory>
#include <vector>

#include "grpc++/grpc++.h"

#include "modules/common/status/status.h"
#include "modules/msgs/hdmap/proto/rpc_service.grpc.pb.h"

namespace roadstar {
namespace common {
namespace hdmap {

class MapClient final {
 public:
  MapClient();

  ::roadstar::common::Status GetLocalMap(
      const ::roadstar::common::PointENU &location, double radius,
      ::roadstar::hdmap::MapProto *const local_map,
      int map_client_timeout_ms = -1);

  ::roadstar::common::Status GetMapElements(
      const std::vector<::roadstar::hdmap::MapUnit> &map_units,
      ::roadstar::hdmap::MapElements *const map_elements,
      int map_client_timeout_ms = -1);

  ::roadstar::common::Status Route(
      const std::vector<::roadstar::common::PointENU> &route_request,
      std::vector<::roadstar::hdmap::MapUnit> *const path,
      int map_client_timeout_ms = -1);

  ::roadstar::common::Status GetRoute(
      std::vector<::roadstar::hdmap::MapUnit> *const path,
      int map_client_timeout_ms = -1);

 private:
  // map rpc service stub
  std::unique_ptr<::roadstar::hdmap::RpcServiceNode::Stub> stub_ = nullptr;
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
