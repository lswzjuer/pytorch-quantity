load("//tools:cpplint.bzl", "cpplint")
load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "remote_visualizer_proto",
    protos = [
        "remote_visualizer.proto",
        "remote_visualizer_grpc.proto",
    ],
    with_grpc = True,
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/msgs/drivers/camera/proto:image_proto",
        "//modules/msgs/drivers/lidar/proto:point_cloud_proto",
    ],
)

cpplint()
