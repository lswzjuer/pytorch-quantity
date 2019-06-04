load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_library(
    name = "gnss_proto",
    deps = [
        ":ins_proto",
        ":gnss_raw_observation_proto",
        ":gnss_best_pose_proto",
        ":config_proto",
        ":imu_proto",
        ":gnss_status_proto",
        ":gnss_proto_proto",
        ":heading_proto",
    ],
)

cc_proto_library(
    name = "ins_proto",
    protos = [
        "ins.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
        ":gnss_best_pose_proto",
    ],
)

py_proto_compile(
    name = "ins_proto_pylib",
    protos = [
        "ins.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
        ":gnss_best_pose_proto_pylib",
    ],
)

cc_proto_library(
    name = "gnss_raw_observation_proto",
    protos = ["gnss_raw_observation.proto"],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "gnss_raw_observation_proto_pylib",
    protos = ["gnss_raw_observation.proto"],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "gnss_best_pose_proto",
    protos = ["gnss_best_pose.proto"],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "gnss_best_pose_proto_pylib",
    protos = ["gnss_best_pose.proto"],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)


cc_proto_library(
    name = "config_proto",
    protos = ["config.proto"],
    deps = [
    ],
)

cc_proto_library(
    name = "imu_proto",
    protos = ["imu.proto"],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "imu_proto_pylib",
    protos = ["imu.proto"],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "gnss_status_proto",
    protos = [
        "gnss_status.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "gnss_status_proto_pylib",
    protos = [
        "gnss_status.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "gnss_proto_proto",
    protos = [
        "gnss.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "gnss_proto_proto_pylib",
    protos = [
        "gnss.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "heading_proto",
    protos = [
        "heading.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "heading_proto_pylib",
    protos = [
        "heading.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "pose_proto",
    protos = ["pose.proto"],
    deps = [
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "pose_proto_pylib",
    protos = ["pose.proto"],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "gps_proto",
    protos = ["gps.proto"],
    deps = [
        ":pose_proto",
        "//modules/common/proto:common_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "gps_proto_pylib",
    protos = ["gps.proto"],
    deps = [
        ":pose_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "corrected_imu_proto",
    protos = ["corrected_imu.proto"],
    deps = [
        ":pose_proto",
        "//modules/common/proto:common_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "corrected_imu_proto_pylib",
    protos = ["corrected_imu.proto"],
    deps = [
        ":pose_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cpplint()
