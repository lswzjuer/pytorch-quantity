load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "adapter_gflags",
    srcs = [
        "adapter_gflags.cc",
    ],
    hdrs = [
        "adapter_gflags.h",
    ],
    deps = [
        "//external:gflags",
    ],
)

cc_library(
    name = "adapter_manager",
    srcs = [
        "adapter_manager.cc",
    ],
    hdrs = [
        "adapter_manager.h",
    ],
    deps = [
        ":adapter_gflags",
        ":message_adapters",
        "//modules/common",
        "//modules/common:log",
        "//modules/common/adapters/proto:adapter_config_proto",
        "//modules/common/message:message_service",
        "//modules/common/transform_listener",
        "//modules/common/util",
        "@ros//:ros_common",
    ],
)

cc_library(
    name = "adapter",
    srcs = [
        "adapter_utils.cc",
    ],
    hdrs = [
        "adapter.h",
        "adapter_utils.h",
    ],
    deps = [
        ":adapter_gflags",
        "//modules/common:log",
        "//modules/common/proto:common_proto",
        "//modules/common/time",
        "//modules/common/util",
        "//modules/msgs/localization/proto:localization_proto",
        "@com_google_protobuf//:protobuf",
        "@ros//:ros_common",
    ],
)

cc_test(
    name = "adapter_test",
    size = "small",
    srcs = [
        "adapter_test.cc",
    ],
    tags = [
        "external",
    ],
    deps = [
        ":adapter",
        ":adapter_gflags",
        ":shared_adapter",
        "//modules/common/adapters/proto:adapter_config_proto",
        "//modules/msgs/localization/proto:localization_proto",
        "@gtest//:main",
        "@ros//:ros_common",
    ],
)

cc_library(
    name = "message_adapters",
    hdrs = [
        "message_adapters.h",
    ],
    deps = [
        ":adapter",
        "//modules/common:log",
        "//modules/common/monitor_log/proto:monitor_log_proto",
        "//modules/msgs/canbus/proto:canbus_proto",
        "//modules/msgs/control/proto:control_proto",
        "//modules/msgs/drivers/asensing/proto:asensing_proto",
        "//modules/msgs/drivers/camera/proto:image_proto",
        "//modules/msgs/drivers/gnss/proto:gnss_ins_proto",
        "//modules/msgs/drivers/lidar/proto:lidar_scan_proto",
        "//modules/msgs/drivers/novatel/proto:gnss_proto",
        "//modules/msgs/drivers/proto:drivers_proto",
        "//modules/msgs/drivers/radar/proto:ultrasonic_radar_proto",
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/msgs/module_conf/proto:system_status_proto",
        "//modules/msgs/perception/proto:perception_proto",
        "//modules/msgs/planning/proto:planning_proto",
    ],
)

cc_library(
    name = "shared_adapter",
    srcs = [
        "shared_adapter.cc",
    ],
    hdrs = [
        "shared_adapter.h",
    ],
    deps = [
        ":adapter",
        ":adapter_manager",
        "//modules/common:macro",
        "//modules/common/util:type_map",
    ],
)

cpplint()
