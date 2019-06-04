load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "map_client",
    srcs = [
        "map_client.cc",
    ],
    hdrs = [
        "map_client.h",
    ],
    deps = [
        "//modules/common:common_gflags",
        "//modules/common:log",
        "//modules/common/status",
        "//modules/msgs/hdmap/proto:hdmap_proto",
        "//modules/msgs/hdmap/proto:hdmap_rpc_proto",
    ],
)

cc_library(
    name = "hdmap_input",
    srcs = ["hdmap_input.cc"],
    hdrs = ["hdmap_input.h"],
    deps = [
        "//modules/common/adapters:adapter_manager",
        "//modules/common/geometry:curve",
        "//modules/common/hdmap_client:map_client",
        "//modules/msgs/hdmap/proto:hdmap_proto",
        "//modules/msgs/localization/proto:localization_proto",
    ],
)

cc_library(
    name = "mock_hdmap_input",
    hdrs = ["mock_hdmap_input.h"],
    deps = [
        ":hdmap_input",
        "@gtest//:main",
    ],
)

cpplint()
