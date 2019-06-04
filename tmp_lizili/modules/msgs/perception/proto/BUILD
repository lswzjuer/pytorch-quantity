package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "perception_proto",
    protos = [
        "camera_config.proto",
        "camera_obstacle.proto",
        "detection_result.proto",
        "fusion_map.proto",
        "geo.proto",
        "grid_map.proto",
        "grid_map_v2.proto",
        "lane_detection.proto",
        "vision_lane.proto",
        "object.proto",
        "obstacle.proto",
        "obstacle_v2.proto",
        "perception_config.proto",
        "radar_filter.proto",
        "time.proto",
        "traffic_light_detection.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:math_proto",
        "//modules/common/proto:sensor_source_proto",
        "//modules/msgs/localization/proto:localization_proto",
    ],
)

py_proto_compile(
    name = "perception_proto_pylib",
    protos = [
        "camera_config.proto",
        "camera_obstacle.proto",
        "detection_result.proto",
        "fusion_map.proto",
        "geo.proto",
        "grid_map.proto",
        "grid_map_v2.proto",
        "lane_detection.proto",
        "vision_lane.proto",
        "object.proto",
        "obstacle.proto",
        "obstacle_v2.proto",
        "perception_config.proto",
        "radar_filter.proto",
        "time.proto",
        "traffic_light_detection.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
        "//modules/msgs/localization/proto:localization_proto_pylib",
    ],
)
