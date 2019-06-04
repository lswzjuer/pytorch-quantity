load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "match",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//modules/integration_test/perception/traffic_light/model:model",
        "//modules/integration_test/perception/common/model:model",
        "//modules/integration_test/perception/util:tool",
        "//modules/common:log",
    ],
    )
cpplint()
