load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "model",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//third_party/json",
        "//modules/common/proto:common_proto",
        "//modules/common/util:util",
        "//modules/common:log",
        "//modules/integration_test/perception/util:tool",
        "@eigen//:eigen",
    ],
)

cpplint()
