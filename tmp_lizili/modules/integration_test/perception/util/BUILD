load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tool",
    srcs = [
        "tool.cc",
    ],
    hdrs = [
        "tool.h",
    ],
    deps = [
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/msgs/perception/proto:perception_proto",
        "//modules/msgs/hdmap/proto:hdmap_proto",
        "//modules/common/coordinate:sensor_coordinate",
        "//modules/common/hdmap_client:map_client",
        "//modules/common:sensor_source",
        "@eigen//:eigen",
    ],
)

cc_library(
    name = "visualization",
    srcs = [
        "visualization.cc",
    ],
    hdrs = [
        "visualization.h"
    ],
    deps = [
        "@opencv3//:common",
        "//modules/integration_test/perception/obstacle/model:model",
    ],
)

cc_library(
    name = "ros_bag_view",
    srcs = [
        "ros_bag_view.cc",
    ],
    hdrs = [
        "ros_bag_view.h"
    ],
    deps = [
        "@ros//:ros_common",
    ],
)

cpplint()
