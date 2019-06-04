load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "point_cloud_proto",
    protos = ["point_cloud.proto"],
)

cc_proto_library(
    name = "common_proto",
    protos = ["common.proto"],
)

cc_proto_library(
    name = "group_service_rpc_proto",
    protos = ["group_service.proto"],
    with_grpc = True,
    deps = ["common_proto"],
)

cc_proto_library(
    name = "bag_service_rpc_proto",
    protos = ["bag_service.proto"],
    with_grpc = True,
    deps = ["common_proto"],
)

cc_proto_library(
    name = "topic_service_rpc_proto",
    protos = ["topic_service.proto"],
    with_grpc = True,
    deps = ["common_proto"],
)

cc_proto_library(
    name = "frame_service_rpc_proto",
    protos = ["frame_service.proto"],
    with_grpc = True,
    deps = ["common_proto"],
)

cc_proto_library(
    name = "task_rpc_proto",
    protos = ["task_service.proto"],
    with_grpc = True,
)

py_proto_compile(
    name = "point_cloud_proto_pylib",
    protos = ["point_cloud.proto"],
)

py_proto_compile(
    name = "common_proto_pylib",
    protos = ["common.proto"],
)

py_proto_compile(
    name = "group_service_rpc_proto_pylib",
    protos = ["group_service.proto"],
    with_grpc = True,
    deps = ["common_proto_pylib"],
)

py_proto_compile(
    name = "bag_service_rpc_proto_pylib",
    protos = ["bag_service.proto"],
    with_grpc = True,
    deps = ["common_proto_pylib"],
)

py_proto_compile(
    name = "topic_service_rpc_proto_pylib",
    protos = ["topic_service.proto"],
    with_grpc = True,
    deps = ["common_proto_pylib"],
)

py_proto_compile(
    name = "frame_service_rpc_proto_pylib",
    protos = ["frame_service.proto"],
    with_grpc = True,
    deps = ["common_proto_pylib"],
)

py_proto_compile(
    name = "task_service_rpc_proto_pylib",
    protos = ["task_service.proto"],
    with_grpc = True,
)
