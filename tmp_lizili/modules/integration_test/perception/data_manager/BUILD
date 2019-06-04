load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "integration_test_data_manager",
    srcs = [
        "integration_test_data_manager.cc",
    ],
    hdrs = [
        "integration_test_data_manager.h",
    ],
    deps = [
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/msgs/perception/proto:perception_proto",
        "//modules/common/util:util",
        "//modules/common:log",
        "//modules/common:macro",
        "//modules/integration_test/perception/proto:integration_test_proto",
        "//modules/integration_test/perception/common/model:model",
        "//modules/integration_test/perception/obstacle/model:model",
        "//modules/integration_test/perception/traffic_light/model:model",
        "//third_party/json",
        "//modules/integration_test/perception/obstacle/serialize:label_data_parser",
        "//modules/integration_test/perception/traffic_light/serialize:label_traffic_light_parser",
        "//modules/integration_test/perception/common:test_object",
    ],
)

cpplint()
