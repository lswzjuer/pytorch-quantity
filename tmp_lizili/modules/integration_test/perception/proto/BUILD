package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

cc_proto_library(
    name = "integration_test_proto",
    protos = [
        "integration_test.proto",
        "perception_test_data_set.proto",
    ],
    deps = [
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/msgs/perception/proto:perception_proto",
    ],
)
