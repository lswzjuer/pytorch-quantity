load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "label_data_parser",
    srcs = [
        "label_data_parser.cc",
    ],
    hdrs = [
        "label_data_parser.h",
        ],
    deps = [
        "//modules/common/util:util",
        "//modules/integration_test/perception/obstacle/model:model",
        "//third_party/json",
        "//modules/common:log",
    ],
)

cc_library(
    name = "obstacle_dump",
    srcs = [
        "obstacle_dump.cc",
        ],
    hdrs = [
        "obstacle_dump.h",
        ],
    deps = [
        "//modules/common/util:util",
        "//modules/common:log",
        "//modules/integration_test/perception/obstacle/model:model",
    ],
)

cpplint()
