load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])



cc_library(
    name = "subnode_callback_manager",
    srcs = [
        "subnode_callback_manager.h",
    ],
    deps = [
        "//modules/common/adapters:adapter_manager",
        "//modules/common:log",
        "//modules/integration_test/common:integration_test_gflags",
        "//modules/integration_test/perception/proto:integration_test_proto",
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/msgs/perception/proto:perception_proto",
        "//modules/integration_test/perception/data_manager:integration_test_data_manager",
        "//modules/integration_test/perception/common/model:model",
    ],
)

cpplint()
