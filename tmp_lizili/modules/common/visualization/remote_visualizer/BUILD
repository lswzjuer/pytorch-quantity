load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "remote_visualizer",
    srcs = [
        "remote_visualizer.cc",
        "remote_visualizer_gflags.cc",
    ],
    hdrs = [
        "remote_visualizer.h",
        "remote_visualizer_gflags.h",
    ],
    deps = [
        "//external:gflags",
        "//modules/common:log",
        "//modules/common/adapters:adapter_manager",
        "//modules/common/visualization/proto",
        "//modules/common/visualization/remote_visualizer/proto:remote_visualizer_proto",
        "@opencv3//:common",
        "@pcl",
    ],
)

cc_binary(
    name = "remote_visualizer_demo",
    srcs = ["remote_visualizer_demo.cc"],
    deps = [
        ":remote_visualizer",
        "//modules/common/util",
    ],
)

cpplint()
