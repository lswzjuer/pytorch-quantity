package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "hdmap_common_proto",
    protos = [
        "hdmap_common.proto",
    ],
)

py_proto_compile(
    name = "hdmap_common_proto_pylib",
    protos = [
        "hdmap_common.proto",
    ],
)

cc_proto_library(
    name = "lanemarker_proto",
    protos = [
        "lanemarker.proto",
    ],
    deps = [
        ":hdmap_common_proto",
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "lanemarker_proto_pylib",
    protos = [
        "lanemarker.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "lane_proto",
    protos = [
        "lane.proto",
    ],
    deps = [
        ":hdmap_common_proto",
        ":lanemarker_proto",
    ],
)

py_proto_compile(
    name = "lane_proto_pylib",
    protos = [
        "lane.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
        ":lanemarker_proto_pylib",
    ],
)

cc_proto_library(
    name = "traffic_light_proto",
    protos = [
        "traffic_light.proto",
    ],
    deps = [
        ":hdmap_common_proto",
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "traffic_light_proto_pylib",
    protos = [
        "traffic_light.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "pavement_proto",
    protos = [
        "pavement.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "pavement_proto_pylib",
    protos = [
        "pavement.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "hdmap_proto",
    protos = [
        "connection.proto",
        "road_edge.proto",
        "section.proto",
    ],
    deps = [
        ":hdmap_common_proto",
        ":lane_proto",
        ":pavement_proto",
        ":traffic_light_proto",
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "hdmap_proto_pylib",
    protos = [
        "connection.proto",
        "road_edge.proto",
        "section.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
        ":lane_proto_pylib",
        ":pavement_proto_pylib",
        ":traffic_light_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "map_proto",
    protos = [
        "map.proto",
    ],
    deps = [
        "hdmap_common_proto",
    ],
)

py_proto_compile(
    name = "map_proto_pylib",
    protos = [
        "map.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
    ],
)

cc_proto_library(
    name = "path_proto",
    protos = [
        "path.proto",
    ],
    deps = [
        ":hdmap_common_proto",
    ],
)

py_proto_compile(
    name = "path_proto_pylib",
    protos = [
        "path.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
    ],
)

cc_proto_library(
    name = "hdmap_config_proto",
    protos = [
        "hdmap_config.proto",
    ],
    deps = [
        ":hdmap_common_proto",
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "hdmap_config_proto_pylib",
    protos = [
        "hdmap_config.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "point_info_proto",
    protos = [
        "point_info.proto",
    ],
    deps = [
        ":hdmap_common_proto",
        "//modules/common/proto:math_proto",
    ],
)

py_proto_compile(
    name = "point_info_proto_pylib",
    protos = [
        "point_info.proto",
    ],
    deps = [
        ":hdmap_common_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "hdmap_rpc_proto",
    protos = [
        "rpc_service.proto",
    ],
    with_grpc = True,
    deps = [
        ":hdmap_config_proto",
        ":hdmap_proto",
        ":map_proto",
        ":path_proto",
        ":point_info_proto",
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "hdmap_rpc_proto_pylib",
    protos = [
        "rpc_service.proto",
    ],
    with_grpc = True,
    deps = [
        ":hdmap_config_proto_pylib",
        ":hdmap_proto_pylib",
        ":path_proto_pylib",
        ":point_info_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)
