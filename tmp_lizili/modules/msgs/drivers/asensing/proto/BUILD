load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_library(
    name = "asensing_proto",
    deps = [
        ":asensing_ins_status_proto",
        ":asensing_ins_proto",
    ],
)

cc_proto_library(
    name = "asensing_ins_status_proto",
    protos = [
        "asensing_ins_status.proto",
    ],
    deps = [
    ],
)

cc_proto_library(
    name = "asensing_ins_proto",
    protos = [
        "asensing_ins.proto",
    ],
    deps = [
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "asensing_ins_proto_pylib",
    protos = [
        "asensing_ins.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cpplint()
