package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
  name = "message_header_proto",
  protos = [
    "message_header.proto",
  ],
)

cc_proto_library(
  name = "diagnose_proto",
  protos = [
    "diagnose.proto",
  ],
  deps = [
    "//modules/common/adapters/proto:adapter_config_proto",
  ],
)

py_proto_compile(
  name = "message_header_py_pb2",
  protos = [
    "message_header.proto",
  ],
)