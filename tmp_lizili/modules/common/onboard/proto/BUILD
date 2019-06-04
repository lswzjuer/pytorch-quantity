package(default_visibility = ["//visibility:public"])
load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "dag_proto",
    protos = [
        "dag_config.proto",
    ],
    deps = [
        "//modules/common/adapters/proto:adapter_config_proto",
    ],
)
