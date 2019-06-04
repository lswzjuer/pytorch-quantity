load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vehicle_manager",
    srcs = [
        "vehicle_manager.cc",
    ],
    hdrs = [
        "vehicle_manager.h",
    ],
    data = [
        "//resources/vehicle_data",
    ],
    deps = [
        "//modules/common:macro",
        "//modules/common/util",
        "//modules/common/vehicle_manager/proto:vehicle_info_proto",
    ],
)

cc_test(
    name = "vehicle_manager_test",
    srcs = [
        "vehicle_manager_test.cc",
    ],
    data = [
        "//resources/vehicle_data",
    ],
    deps = [
        ":vehicle_manager",
        "@gtest//:main",
    ],
)
