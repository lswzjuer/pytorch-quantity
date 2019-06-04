load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "local_map",
    srcs = [
        "local_map.cc",
    ],
    hdrs = [
        "local_map.h",
    ],
    deps = [
        "//modules/common/hdmap/base:map",
        "//modules/common/hdmap/client:map_client",
    ],
)

cpplint()
