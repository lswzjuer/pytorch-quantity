package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "simulation_world_proto",
    protos = ["simulation_world.proto"],
    deps = [
        "//modules/common/monitor_log/proto:monitor_log_proto",
        "//modules/msgs/canbus/proto:canbus_proto",
        "//modules/msgs/perception/proto:perception_proto",
        "//modules/msgs/planning/proto:planning_proto",
    ],
)

cc_proto_library(
    name = "hmi_config_proto",
    protos = ["hmi_config.proto"],
)

cc_proto_library(
    name = "hw_monitor_result_proto",
    protos = ["hw_monitor_result.proto"],
)

cc_proto_library(
    name = "camera_data_proto",
    protos = ["camera_data.proto"],
)

cc_proto_library(
    name = "local_map_proto",
    protos = ["local_map.proto"],
    deps = [
        "//modules/msgs/hdmap/proto:hdmap_proto",
        "//modules/msgs/hdmap/proto:hdmap_rpc_proto",
    ],
)

cc_proto_library(
    name = "available_maps_proto",
    protos = ["available_maps.proto"],
    deps = [
        "//modules/msgs/hdmap/proto:hdmap_config_proto",
    ],
)

cc_proto_library(
    name = "response_proto",
    protos = ["response.proto"],
    deps = [
        "//modules/msgs/dreamview/proto:available_maps_proto",
        "//modules/msgs/dreamview/proto:camera_data_proto",
        "//modules/msgs/dreamview/proto:hmi_config_proto",
        "//modules/msgs/dreamview/proto:hmi_status_proto",
        "//modules/msgs/dreamview/proto:local_map_proto",
        "//modules/msgs/dreamview/proto:player_proto",
        "//modules/msgs/dreamview/proto:rosbag_readme_proto",
        "//modules/msgs/dreamview/proto:simulation_world_proto",
        "//modules/msgs/hdmap/proto:hdmap_proto",
        "//modules/msgs/planning/proto:planning_debug_info_proto",
    ],
)

cc_proto_library(
    name = "hmi_status_proto",
    protos = ["hmi_status.proto"],
    deps = [
        "//modules/msgs/module_conf/proto:system_status_proto",
    ],
)

cc_proto_library(
    name = "point_cloud_proto",
    protos = ["point_cloud.proto"],
)

cc_proto_library(
    name = "rosbag_readme_proto",
    protos = ["rosbag_readme.proto"],
)

cc_proto_library(
    name = "player_proto",
    protos = ["player.proto"],
)
