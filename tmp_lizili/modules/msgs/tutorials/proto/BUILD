package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "rpc_service_proto",
    protos = ["rpc_service.proto"],
    with_grpc = True,
)

py_proto_compile(
    name = "rpc_service_proto_pylib",
    protos = ["rpc_service.proto"],
    with_grpc = True,
)
