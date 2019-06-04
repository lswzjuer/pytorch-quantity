package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "planning_proto",
    protos = [
        "planning.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "planning_proto_pylib",
    protos = [
        "planning.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "planning_debug_info_proto",
    protos = [
        "planning_debug_info.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "planning_debug_info_proto_pylib",
    protos = [
        "planning_debug_info.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)
