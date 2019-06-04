load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kvaser_can_client",
    srcs = [
        "kvaser_can_client.cc",
    ],
    hdrs = [
        "kvaser_can_client.h",
    ],
    deps = [
        "//modules/common/canbus/can_client",
    ],
    linkopts = ["-lcanlib"],
)

cc_test(
    name = "kvaser_can_client_test",
    size = "small",
    srcs = [
        "kvaser_can_client_test.cc",
    ],
    deps = [
        "//modules/common/canbus/can_client/kvaser:kvaser_can_client",
        "//modules/common/canbus/common:canbus_common",
        "//modules/common:log",
        "@gtest//:main",
    ],
)

cpplint()
