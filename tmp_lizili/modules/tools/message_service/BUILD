load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "extract_msg",
    srcs = [
        "extract_msg.cc",
    ],
    deps = [
        "//external:gflags",
        "//modules/common:log",
        "//modules/common:macro",
        "//modules/common/adapters/proto:adapter_config_proto",
        "//modules/common/message/proto:message_header_proto",
        "//modules/common/message/tools:message_bag",
        "//modules/common/message/tools:utils",
    ],
)
