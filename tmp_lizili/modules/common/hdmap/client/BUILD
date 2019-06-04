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

cpplint()
