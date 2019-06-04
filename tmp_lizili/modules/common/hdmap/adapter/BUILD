load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "local_map_manager",
    srcs = [
        "local_map_manager.cc",
    ],
    hdrs = [
        "local_map_manager.h",
    ],
    deps = [
        "//modules/common:macro",
        "//modules/common/hdmap/common:local_map",
        "//modules/common/util:thread_pool",
    ],
)

cc_library(
    name = "route_manager",
    srcs = [
        "route_manager.cc",
    ],
    hdrs = [
        "route_manager.h",
    ],
    deps = [
        "//modules/common:log",
        "//modules/common:macro",
        "//modules/common/hdmap/base:map_unit_hash",
        "//modules/common/hdmap/client:map_client",
        "//modules/common/util:thread_pool",
    ],
)

cpplint()
