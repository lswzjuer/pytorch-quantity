load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "traffic_light_reporter",
    srcs = [
        "traffic_light_reporter.cc",
        ],
    hdrs = [
        "traffic_light_reporter.h",
        ],
    deps = [
        "//modules/common/util:util",
        "//modules/common:log",
        "//modules/integration_test/perception/traffic_light/match:match",
        "//modules/integration_test/perception/traffic_light/model:model",
        "//modules/integration_test/perception/common/model:model",
        "//modules/integration_test/perception/data_manager:integration_test_data_manager",
    ],
)

cpplint()
