load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "spline",
    srcs = [
        "spline.cc"
    ],
    hdrs = [
        "spline.h"
    ],
    deps = [

    ]
)


cc_library(
    name = "graph",
    srcs = [
        "graph.cc"
    ],
    hdrs = [
        "graph.h"
    ],
    deps = [
        	"//modules/common:log"
    ]
)
cc_library(
    name = "curve",
    srcs = [
        "curve.cc",
    ],
    hdrs = [
        "curve.h"
    ],
    deps = [
        ":spline"
    ]
)

cc_library(
    name = "lane_region",
    srcs = [
        "lane_region.cc"
    ],
    hdrs = [
        "lane_region.h"
    ],
    deps = [
        ":curve",
        "//modules/common:log",
        "//modules/common/proto:common_proto", 
        "@opencv3//:common",
    ]
)

cc_test(
    name = "lane_region_test",
    size = "small",
    srcs = [
        "lane_region_test.cc",
    ],
    deps = [
        ":lane_region",
        "@gtest//:main",
    ],
)

cc_test(
    name = "curve_test",
    size = "small",
    srcs = [
        "curve_test.cc",
    ],
    deps = [
        ":curve",
        "@gtest//:main",
    ],
)

cpplint()
