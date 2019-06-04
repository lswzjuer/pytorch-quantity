#include "modules/common/hdmap_client/map_client.h"

#include <utility>

#include "google/protobuf/util/json_util.h"
#include "modules/common/common_gflags.h"
#include "modules/common/log.h"

namespace roadstar {
namespace common {

using common::Polygon;
using google::protobuf::RepeatedPtrField;
using grpc::Channel;
using grpc::ClientContext;

using common::Polygon;
using hdmap::GetAvailableMapsRequest;
using hdmap::GetAvailableMapsResponse;
using hdmap::GetCurrMapRequest;
using hdmap::GetCurrMapResponse;
using hdmap::GetCurrRouteRequest;
using hdmap::GetCurrRouteResponse;
using hdmap::GetLocalMapRequest;
using hdmap::GetLocalMapResponse;
using hdmap::GetLocalPathRequest;
using hdmap::GetLocalPathResponse;
using hdmap::GetMapElementsRequest;
using hdmap::GetMapElementsResponse;
using hdmap::GetMapWarningsRequest;
using hdmap::GetMapWarningsResponse;
using hdmap::GetPointOnRoadRequest;
using hdmap::GetPointOnRoadResponse;
using hdmap::GetRoutingPathRequest;
using hdmap::GetRoutingPathResponse;
using hdmap::Lanemarker;
using hdmap::MapConfig;
using hdmap::MapElements;
using hdmap::MapProto;
using hdmap::MapUnit;
using hdmap::Path;
using hdmap::RpcServiceNode;
using hdmap::SetMapRequest;
using hdmap::SetMapResponse;
using hdmap::SetRouteRequest;
using hdmap::SetRouteResponse;
using hdmap::SetRoutingPointsRequest;
using hdmap::SetRoutingPointsResponse;

namespace {

constexpr int kMapBuffSize = 20;

void FillClientContext(const int map_client_timeout_ms,
                       ClientContext* context) {
  if (map_client_timeout_ms > 0) {
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(map_client_timeout_ms);
    context->set_deadline(deadline);
  }
}

}  // namespace

MapClient::MapClient(
    std::unique_ptr<roadstar::hdmap::RpcServiceNode::Stub> stub) {
  stub_ = std::move(stub);
}

MapClient::MapClient() {
  auto channel = grpc::CreateChannel(FLAGS_hdmap_rpc_service_address,
                                     grpc::InsecureChannelCredentials());
  stub_ = RpcServiceNode::NewStub(channel);
}

Status MapClient::GetLocalMap(const double x, const double y,
                              const double radius, MapProto* local_map,
                              const int map_client_timeout_ms) const {
  // Container for sending to server
  GetLocalMapRequest request;
  request.mutable_location()->set_x(x);
  request.mutable_location()->set_y(y);
  request.set_radius(radius);

  // Container for the map element ids
  GetLocalMapResponse response;

  // Context for client. It could be used to convey extra information
  // to the server
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetLocalMap(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  *local_map = std::move(*response.mutable_local_map());
  return Status::OK();
}

Status MapClient::GetLocalPath(const double x, const double y,
                               const double forward_distance,
                               const double backward_distance,
                               roadstar::hdmap::Path* path,
                               const int map_client_timeout_ms) {
  // Container for sending to server
  GetLocalPathRequest request;
  request.mutable_location()->set_x(x);
  request.mutable_location()->set_y(y);
  request.set_forward_distance(forward_distance);
  request.set_backward_distance(backward_distance);

  GetLocalPathResponse response;
  // Context for client. It could be used to convey extra information
  // to the server
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetLocalPath(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  *path = std::move(*response.mutable_local_path());
  return Status::OK();
}

Status MapClient::RetrieveMapElements(
    const RepeatedPtrField<MapUnit>& map_units,
    const int map_client_timeout_ms) {
  // Container for sending to server.
  GetMapElementsRequest request;
  for (auto map_unit : map_units) {
    if (map_unit.type() == MapUnit::MAP_UNIT_SECTION &&
        sections_.find(map_unit.id()) == sections_.end()) {
      *request.add_map_units() = map_unit;
    } else if (map_unit.type() == MapUnit::MAP_UNIT_CONNECTION &&
               connections_.find(map_unit.id()) == connections_.end()) {
      *request.add_map_units() = map_unit;
    }
  }

  if (request.map_units_size() == 0) {
    return Status::OK();
  }

  // Container for the map elements data.
  GetMapElementsResponse response;

  // Context for client. It could be used to convey extra information
  // to the server.
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetMapElements(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  mutex_.lock();
  // Remove some expired map unit.
  while (!cached_map_units_.empty() &&
         cached_map_units_.size() + request.map_units_size() > kMapBuffSize) {
    auto map_unit = cached_map_units_.back();
    cached_map_units_.pop_back();
    if (map_unit.type() == MapUnit::MAP_UNIT_SECTION) {
      sections_.erase(map_unit.id());
    } else if (map_unit.type() == MapUnit::MAP_UNIT_CONNECTION) {
      connections_.erase(map_unit.id());
    }
  }

  // Insert the new added map unit.
  for (const auto& section_with_id : response.map_elements().sections()) {
    sections_[section_with_id.first] = section_with_id.second;
  }
  for (const auto& conn_with_id : response.map_elements().connections()) {
    connections_[conn_with_id.first] = conn_with_id.second;
  }
  mutex_.unlock();

  return Status::OK();
}

Status MapClient::GetLocalBoundary(const double x, const double y,
                                   const double radius,
                                   std::vector<Polygon>* polygons,
                                   const int map_client_timeout_ms) {
  MapProto local_map;
  auto status = GetLocalMap(x, y, radius, &local_map, map_client_timeout_ms);
  if (!status.ok()) {
    AERROR << "Call GetLocalMap fail: " << x << " " << y << " "
           << status.error_message();
    return status;
  }
  if (local_map.map_units_size() > 0) {
    status = RetrieveMapElements(local_map.map_units(), map_client_timeout_ms);
    if (!status.ok()) {
      AERROR << "Call RetrieveMapElement fail: " << x << " " << y << " "
             << status.error_message();
      return status;
    }
  }

  polygons->clear();
  polygons->reserve(local_map.map_units_size());
  for (const auto& map_unit : local_map.map_units()) {
    if (map_unit.type() == MapUnit::MAP_UNIT_SECTION) {
      polygons->emplace_back(sections_[map_unit.id()].polygon());
    } else if (map_unit.type() == MapUnit::MAP_UNIT_CONNECTION) {
      polygons->emplace_back(connections_[map_unit.id()].polygon());
    }
  }

  return Status::OK();
}

Status MapClient::GetLocalLanemarkers(const double x, const double y,
                                      const double forward_distance,
                                      const double backward_distance,
                                      std::vector<Lanemarkers>* lanemarkers,
                                      const int map_client_timeout_ms) {
  Path path;
  auto status = GetLocalPath(x, y, forward_distance, backward_distance, &path,
                             map_client_timeout_ms);

  if (!status.ok()) {
    return status;
  }
  status = RetrieveMapElements(path.path_units(), map_client_timeout_ms);

  lanemarkers->clear();
  lanemarkers->reserve(path.path_units_size());
  for (const auto& path_unit : path.path_units()) {
    if (path_unit.type() == MapUnit::MAP_UNIT_SECTION) {
      Lanemarkers lms;
      for (const auto& lm : sections_[path_unit.id()].lanemarkers()) {
        lms.emplace_back(lm);
      }
      lanemarkers->emplace_back(std::move(lms));
    }
  }

  return Status::OK();
}

Status MapClient::SetMap(const std::string& map_name,
                         const int map_client_timeout_ms) {
  SetMapRequest request;
  request.set_map(map_name);

  SetMapResponse response;

  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->SetMap(&context, request, &response);
  mutex_.lock();
  cached_map_units_.clear();
  sections_.clear();
  connections_.clear();
  mutex_.unlock();

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  return Status::OK();
}

Status MapClient::GetCurrMap(std::string* const curr_map_name,
                             const int map_client_timeout_ms) {
  GetCurrMapRequest request;

  GetCurrMapResponse response;

  // Context for client. It could be used to convey extra information
  // to the server
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetCurrMap(&context, request, &response);

  if (!status.ok()) {
    curr_map_name->clear();
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  *curr_map_name = response.map_name();
  return Status::OK();
}

Status MapClient::SetRoute(const std::string& route_name,
                           const int map_client_timeout_ms) {
  SetRouteRequest request;
  request.set_route_name(route_name);

  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  SetRouteResponse response;
  grpc::Status status = stub_->SetRoute(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  return Status::OK();
}

Status MapClient::SetRoutingPoints(const std::vector<PointENU>& routing_points,
                                   const int map_client_timeout_ms) {
  SetRoutingPointsRequest request;
  for (const auto& point : routing_points) {
    *request.add_points() = point;
  }

  SetRoutingPointsResponse response;

  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->SetRoutingPoints(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  return Status::OK();
}

Status MapClient::GetRoutingPath(roadstar::hdmap::Path* routing_path,
                                 const int map_client_timeout_ms) {
  GetRoutingPathRequest request;
  GetRoutingPathResponse response;

  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetRoutingPath(&context, request, &response);
  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  *routing_path = response.path();

  return Status::OK();
}

Status MapClient::GetCurrRoute(std::string* const curr_route_name,
                               const int map_client_timeout_ms) {
  GetCurrRouteRequest request;

  GetCurrRouteResponse response;

  // Context for client. It could be used to convey extra information
  // to the server
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetCurrRoute(&context, request, &response);

  if (!status.ok()) {
    curr_route_name->clear();
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  *curr_route_name = response.route_name();
  return Status::OK();
}

Status MapClient::RetrieveMapElements(
    const std::vector<roadstar::hdmap::MapUnit>& map_units,
    roadstar::hdmap::MapElements* map_elements,
    const int map_client_timeout_ms) {
  auto status = RetrieveMapElements({map_units.begin(), map_units.end()},
                                    map_client_timeout_ms);

  if (!status.ok()) {
    return status;
  }

  for (const auto map_unit : map_units) {
    if (map_unit.type() == MapUnit::MAP_UNIT_SECTION) {
      map_elements->mutable_sections()->insert(
          {map_unit.id(), sections_[map_unit.id()]});
    } else if (map_unit.type() == MapUnit::MAP_UNIT_CONNECTION) {
      map_elements->mutable_connections()->insert(
          {map_unit.id(), connections_[map_unit.id()]});
    }
  }

  return Status::OK();
}

Status MapClient::GetAvailableMaps(
    std::unordered_map<std::string, MapConfig>* available_maps,
    const int map_client_timeout_ms) {
  GetAvailableMapsRequest request;
  GetAvailableMapsResponse response;

  // Context for client. It could be used to convey extra information
  // to the server
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetAvailableMaps(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }
  for (const auto& kval : response.available_maps()) {
    available_maps->insert({kval.first, kval.second});
  }

  return Status::OK();
}

Status MapClient::GetPointOnRoad(
    const std::vector<PointENU>& points, const PointENU& localization,
    std::vector<roadstar::hdmap::PointInfo>* const point_infos, double radius,
    const int map_client_timeout_ms) {
  GetPointOnRoadRequest request;
  for (const auto& point : points) {
    *request.add_points() = point;
  }
  request.set_radius(radius);
  *request.mutable_location() = localization;
  GetPointOnRoadResponse response;

  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetPointOnRoad(&context, request, &response);
  if (!status.ok()) {
    AERROR << "MapClient::GetPointOnRoad hdmap error.Error message = "
           << status.error_message();
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }
  for (const auto it : response.point_infos()) {
    point_infos->push_back(it);
  }
  return Status::OK();
}

}  // namespace common
}  // namespace roadstar
