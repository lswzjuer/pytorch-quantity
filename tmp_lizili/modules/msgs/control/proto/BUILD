package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "control_proto",
    protos = [
        "control_command.proto",
        "control_status.proto",
        "pad_msg.proto",
    ],
    deps = [
	":control_debug_proto",
        "//modules/msgs/canbus/proto:canbus_proto",
        "//modules/common/proto:common_proto",
    ],
)

py_proto_compile(
    name = "control_proto_pylib",
    protos = [
        "control_command.proto",
        "control_status.proto",
        "pad_msg.proto",
    ],
    deps = [
	":control_debug_pylib",
        "//modules/msgs/canbus/proto:canbus_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "control_debug_proto",
    protos = [
        "control_debug.proto",
        "speed_controller_debug.proto",
        "braking_controller_debug.proto",
        "steer_controller_debug.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        "//modules/msgs/planning/proto:planning_proto",
    ],
)

py_proto_compile(
    name = "control_debug_pylib",
    protos = [
        "control_debug.proto",
        "speed_controller_debug.proto",
        "braking_controller_debug.proto",
        "steer_controller_debug.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
        "//modules/msgs/planning/proto:planning_proto_pylib",
    ],
)

cc_proto_library(
    name = "state_space_proto",
    protos = [
        "state_space.proto",
    ],
    deps = [
        "//modules/common/proto:math_proto",
    ],
)

py_proto_compile(
    name = "state_space_pylib",
    protos = [
        "state_space.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
    ],
)

cc_proto_library(
    name = "closed_loop_model_proto",
    protos = [
        "closed_loop_model.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto",
        ":state_space_proto",
    ],
)

py_proto_compile(
    name = "closed_loop_model_pylib",
    protos = [
        "closed_loop_model.proto",
    ],
    deps = [
        "//modules/common/proto:common_proto_pylib",
        ":state_space_pylib",
    ],
)
