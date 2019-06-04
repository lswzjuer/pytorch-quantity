load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])



cc_binary(
    name = "report_compare",
    srcs = [
        "report_compare.cc",
    ],
    deps = [
        "//modules/integration_test/perception/obstacle/model:model",
        "//modules/common:log",
    ],
)

cc_library(
    name = "obstacle_reporter",
    srcs = [
        "obstacle_reporter.cc",
        ],
    hdrs = [
        "obstacle_reporter.h",
        ],
    deps = [
        "//modules/common/util:util",
        "//modules/common:log",
        "//modules/integration_test/perception/obstacle/match:match",
        "//modules/integration_test/perception/obstacle/model:model",
        "//modules/integration_test/perception/common/model:model",
        "//modules/integration_test/perception/obstacle/serialize:obstacle_dump",
    ],
)

cpplint()
