load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xml_param_reader",
    srcs = [
        "xml_param_reader.cc",
    ],
    hdrs = [
        "xml_param_reader.h",
    ],
    linkopts = [
        "-lpugixml",
    ],
    deps = [
        "//modules/common:log",
        "//modules/integration_test/perception/common/model:model",
    ],
)

cc_binary(
    name = "get_bag_parameter",
    srcs = [
        "get_bag_parameter.cc",
    ],
    deps = [
        ":xml_param_reader",
        "//modules/common:log",
    ],
)

cpplint()
