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
        "//modules/integration_test/perception/obstacle/model:model",
        "//modules/integration_test/perception/common/model:model",
        "//modules/integration_test/perception/util:visualization",
        "//modules/integration_test/perception/util:tool",
        "//modules/msgs/perception/proto:perception_proto",
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/common/hdmap_client:map_client",
        "//modules/common/util:util",
        "//modules/common:log",
    ],
    )
cpplint()
