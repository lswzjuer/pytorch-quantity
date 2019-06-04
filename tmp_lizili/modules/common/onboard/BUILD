load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "onboard",
    srcs = [
        "dag_streaming.cc",
        "subnode.cc",
        "subnode_helper.cc",
    ],
    hdrs = [
        "dag_streaming.h",
        "subnode.h",
        "subnode_helper.h",
    ],
    deps = [
        "//modules/common",
        "//modules/common:log",
        "//modules/common/util:registerer",
        "//modules/common/util:thread",
        "//modules/common/adapters:adapter_manager",
        "//modules/common/status",
        "//modules/common/time",
        "//modules/common/onboard/proto:dag_proto",
        "//modules/common/coordinate:sensor_coordinate",
        "@eigen",
        "@ros//:ros_common",
    ],
)

cc_test(
    name = "subnode_test",
    size = "small",
    srcs = [
        "subnode_test.cc",
    ],
    data = [
        "//resources/perception:perception_data",
    ],
    deps = [
        ":onboard",
        "//external:gflags",
        "//modules/common:log",
        "@gtest",
        "@gtest//:main",
    ],
)

cpplint()
