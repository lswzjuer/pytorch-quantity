package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
  name = "message_bag_proto",
  protos = [
    "message_bag.proto",
  ],
)

py_proto_compile(
  name = "message_bag_proto_py",
  protos = [
    "message_bag.proto",
  ],
)