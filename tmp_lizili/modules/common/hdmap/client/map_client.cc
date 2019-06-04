#include "modules/common/hdmap/client/map_client.h"

#include <utility>

#include "modules/common/common_gflags.h"

namespace roadstar {
namespace common {
namespace hdmap {

using grpc::ClientContext;

namespace {

void FillClientContext(int map_client_timeout_ms, ClientContext *context) {
  if (map_client_timeout_ms > 0) {
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(map_client_timeout_ms);
    context->set_deadline(deadline);
  }
}

}  // namespace

MapClient::MapClient() {
  auto channel = grpc::CreateChannel(FLAGS_hdmap_rpc_service_address,
                                     grpc::InsecureChannelCredentials());
  stub_ = ::roadstar::hdmap::RpcServiceNode::NewStub(channel);
}

Status MapClient::GetLocalMap(const PointENU &location, double radius,
                              ::roadstar::hdmap::MapProto *const local_map,
                              int map_client_timeout_ms) {
  // Container for sending to server
  ::roadstar::hdmap::GetLocalMapRequest request;
  *(request.mutable_location()) = location;
  request.set_radius(radius);

  // Container for the map element ids
  ::roadstar::hdmap::GetLocalMapResponse response;

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

Status MapClient::GetMapElements(
    const std::vector<::roadstar::hdmap::MapUnit> &map_units,
    ::roadstar::hdmap::MapElements *const map_elements,
    int map_client_timeout_ms) {
  // Return ok if the request map unit size is zero.
  if (map_units.size() == 0) {
    return Status::OK();
  }

  // Construct the rpc request.
  ::roadstar::hdmap::GetMapElementsRequest request;
  for (auto map_unit : map_units) {
    *request.add_map_units() = map_unit;
  }

  // Container for the map elements data.
  ::roadstar::hdmap::GetMapElementsResponse response;

  // Context for client. It could be used to convey extra information
  // to the server.
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  grpc::Status status = stub_->GetMapElements(&context, request, &response);

  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  *map_elements = std::move(*response.mutable_map_elements());

  return Status::OK();
}

Status MapClient::Route(
    const std::vector<::roadstar::common::PointENU> &route_request,
    std::vector<::roadstar::hdmap::MapUnit> *const path,
    int map_client_timeout_ms) {
  ::roadstar::hdmap::SetRoutingPointsRequest request;
  for (const auto &pt : route_request) {
    *request.add_points() = pt;
  }
  // Context for client. It could be used to convey extra information
  // to the server.
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  ::roadstar::hdmap::SetRoutingPointsResponse response;
  auto status = stub_->SetRoutingPoints(&context, request, &response);
  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }
  for (auto &map_unit : response.path().path_units()) {
    path->push_back(std::move(map_unit));
  }

  return Status::OK();
}

Status MapClient::GetRoute(std::vector<::roadstar::hdmap::MapUnit> *const path,
                           int map_client_timeout_ms) {
  ::roadstar::hdmap::GetRoutingPathRequest request;

  // Context for client. It could be used to convey extra information
  // to the server.
  ClientContext context;
  FillClientContext(map_client_timeout_ms, &context);

  ::roadstar::hdmap::GetRoutingPathResponse response;
  auto status = stub_->GetRoutingPath(&context, request, &response);
  if (!status.ok()) {
    return Status(ErrorCode::HDMAP_ERROR, status.error_message());
  }

  for (auto &map_unit : response.path().path_units()) {
    path->push_back(std::move(map_unit));
  }

  return Status::OK();
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
