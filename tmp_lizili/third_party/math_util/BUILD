# Get rotation and transition matrix between two sets of points
load("//tools:cpplint.bzl", "cpplint")
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "get_rt_matrix",
    srcs = [
        "get_rt_matrix.cc",
    ],
    hdrs = [
        "get_rt_matrix.h",
    ],
    deps = [
        "@opencv3//:common",
    ],
)