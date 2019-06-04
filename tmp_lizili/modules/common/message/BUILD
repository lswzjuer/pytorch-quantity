load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "message_service",
    srcs = [
        "message_service.cc",
        "message_sender.cc",
        "message_receiver.cc",
    ],
    hdrs = [
        "message_service.h",
        "message_sender.h",
        "message_receiver.h",
    ],
    deps = [
        ":utils",
        "//external:gflags",
        "//modules/common",
        "//modules/common:log",
        "//modules/common/adapters/proto:adapter_config_proto",
        "//modules/common/message/tools:utils",
        "//modules/common/message/proto:diagnose_proto",
        "//modules/common/message/proto:message_header_proto",
        "//modules/common/module_conf:module_util",
        "//modules/common/util",
        "//modules/msgs/module_conf/proto:module_conf_proto",
        "@ros//:ros_common",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//modules/common:log",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "utils_test",
    size = "small",
    srcs = [
        "utils_test.cc",
    ],
    deps = [
        ":utils",
        "//modules/common/message/proto:message_header_proto",
        "@gtest//:main",
    ],
)

cc_test(
    name = "message_service_test",
    size = "small",
    srcs = [
        "message_service_test.cc",
    ],
    data = [
        ":message_service_test_data",
    ],
    deps = [
        ":message_service",
        ":utils",
        "//external:gflags",
        "//modules/common/message/proto:message_header_proto",
        "@com_google_protobuf//:protobuf",
        "@gtest//:main",
    ],
)

filegroup(
    name = "message_service_test_data",
    srcs = glob([
        "testdata/**",
    ]),
)
cpplint()
