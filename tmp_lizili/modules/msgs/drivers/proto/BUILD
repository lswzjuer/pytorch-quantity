load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "coords_calibration_proto",
    protos = ["coords_calibration.proto"],
)

py_proto_compile(
    name = "coords_calibration_pylib",
    protos = ["coords_calibration.proto"],
)

cc_proto_library(
    name = "drivers_proto",
    protos = [
        "conti_radar.proto",
        "conti_radar_conf.proto",
        "delphi_esr.proto",
        "esr.proto",
        "gps.proto",
        "pluto_imu.proto",
        "rsds.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/common/proto:error_code_proto",
        "//modules/common/proto:header_proto",
        "//modules/msgs/canbus/proto:canbus_proto",
        "//modules/msgs/drivers/novatel/proto:gnss_proto",
    ],
)

py_proto_compile(
    name = "drivers_proto_pylib",
    protos = [
        "conti_radar.proto",
        "conti_radar_conf.proto",
        "delphi_esr.proto",
        "esr.proto",
        "gps.proto",
        "pluto_imu.proto",
        "rsds.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
        "//modules/msgs/canbus/proto:canbus_proto_pylib",
    ],
)

cc_proto_library(
    name = "radar_install_proto",
    protos = [
        "conti_calibration.proto",
        "radar_install.proto",
    ],
)

py_proto_compile(
    name = "radar_install_pylib",
    protos = [
        "conti_calibration.proto",
        "radar_install.proto",
    ],
)
