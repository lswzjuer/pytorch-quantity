load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "statistical_analysis",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//modules/common:log",
        "//modules/integration_test/perception/obstacle/serialize:label_data_parser",
        "//modules/integration_test/perception/common/model:model",
        "//modules/integration_test/perception/obstacle/model:model",
        "//modules/integration_test/perception/obstacle/report:obstacle_reporter",
        "//modules/integration_test/perception/traffic_light/report:traffic_light_reporter",
        "//modules/integration_test/perception/data_manager:integration_test_data_manager",
        "//modules/integration_test/perception/common:test_object",
    ],
    )
cpplint()
