package(default_visibility = ["//visibility:public"])

load("//tools:cpplint.bzl", "cpplint")
load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "proto",
    protos = glob(["*.proto"]),
    deps = [
        "//modules/common/proto:colormap_proto",
    ],
)

cpplint()
