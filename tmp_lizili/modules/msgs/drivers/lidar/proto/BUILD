load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "lidar_scan_proto",
    protos = [
        "lidar_scan.proto",
    ],
    deps = [
        "//modules/common/proto:header_proto",
    ],
)

cc_proto_library(
    name = "point_cloud_proto",
    protos = [
        "point_cloud.proto",
    ],
    deps = [
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "lidar_proto_pylib",
    protos = [
        "lidar_scan.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cpplint()
