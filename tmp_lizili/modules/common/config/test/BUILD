load("//tools:cpplint.bzl", "cpplint")
load("//tools:config_library.bzl", "config_library")

package(default_visibility = ["//visibility:public"])

config_library(
    name = "config_test_config",
    messages = [
        "common::ConfigTest",
        "common::ConfigTest2",
    ],
    protos = [
        "config_test.proto",
    ],
)

cc_test(
    name = "config_test",
    srcs = [
        "config_test.cc",
    ],
    data = [
        "configs",
    ],
    deps = [
        ":config_test_config",
        "//modules/common/config:config_manager",
        "@gtest",
        "@gtest//:main",
    ],
)

cpplint()
