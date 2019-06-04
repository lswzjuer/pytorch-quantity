load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "perception_test",
    srcs = [
        "main.cc",
    ],
    linkopts = [
        "-fopenmp",
    ],
    deps = [
        ":perception_test_lib",
        "//modules/common/util",
        "@ros//:ros_common",
    ],
)

cc_library(
    name = "perception_test_lib",
    srcs = [
        "perception_test.cc",
    ],
    hdrs = [
        "perception_test.h",
    ],
    deps = [
        "//modules/common:roadstar_app",
        "//modules/integration_test/common/xml_param:xml_param_reader",
        "//modules/integration_test/perception/callback_manager:subnode_callback_manager",
        "//modules/integration_test/perception/common/model",
        "//modules/integration_test/perception/obstacle/model",
        "//modules/integration_test/perception/statistical_analysis",
        "@ros//:ros_common",
    ],
)

cpplint()
