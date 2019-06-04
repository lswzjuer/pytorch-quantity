load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_compile")

cc_proto_library(
    name = "module_conf_proto",
    protos = [
        "module_conf.proto",
    ],
    deps = [
        "//modules/common/adapters/proto:adapter_config_proto",
    ],
)

py_proto_compile(
    name = "module_conf_proto_pylib",
    protos = [
        "module_conf.proto",
    ],
    deps = [
        "//modules/common/adapters/proto:adapter_config_proto_pylib",
    ],
)

cc_proto_library(
    name = "system_status_proto",
    protos = [
        "system_status.proto",
    ],
    deps = [
        ":module_conf_proto",
        "//modules/common/proto:header_proto",
    ],
)

py_proto_compile(
    name = "system_status_proto_pylib",
    protos = [
        "system_status.proto",
    ],
    deps = [
        ":module_conf_proto_pylib",
        "//modules/common/proto:common_proto_pylib",
    ],
)

cpplint()
