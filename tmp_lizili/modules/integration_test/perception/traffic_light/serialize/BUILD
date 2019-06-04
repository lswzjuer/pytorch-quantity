load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "label_traffic_light_parser",
    srcs = [
        "label_traffic_light_parser.cc",
    ],
    hdrs = [
        "label_traffic_light_parser.h",
        ],
    deps = [
        "//modules/common/util:util",
        "//modules/integration_test/perception/traffic_light/model:model",
        "//third_party/json",
        "//modules/common:log",
    ],
)

cpplint()
