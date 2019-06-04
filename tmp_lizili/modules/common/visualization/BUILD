load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//modules/common/adapters:adapter_manager",
        "//modules/common:log",
        "//modules/common/util:colormap",
        "//modules/common/coordinate:sensor_coordinate",
        "//modules/common/visualization/proto",
        "//modules/common/hdmap_client:hdmap_input",
        "@opencv3//:common",
        "@eigen",
    ],
)

cpplint()
