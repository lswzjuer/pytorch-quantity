package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "calibration_proto",
    protos = [
        "calibration_config.proto",
        "transform.proto",
        "camera_conf.proto",
    ],
    deps = [
        "//modules/common/proto:sensor_source_proto",
    ],
)
