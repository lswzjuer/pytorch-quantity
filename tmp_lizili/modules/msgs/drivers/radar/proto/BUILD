load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "ultrasonic_radar_proto",
    protos = [
        "ultrasonic_radar.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "ultrasonic_radar_proto_pylib",
    protos = [
        "ultrasonic_radar.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cpplint()
