package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "vehicle_info_proto",
    protos = [
        "vehicle_info.proto",
    ],
)

py_proto_compile(
    name = "vehicle_info_pylib",
    protos = [
        "vehicle_info.proto",
    ],
)
