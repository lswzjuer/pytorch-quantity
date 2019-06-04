load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "can_dump",
    srcs = ["can_dump.cc"],
    deps = [
        "//external:gflags",
        "//modules/common/canbus/can_client:can_client_factory",
        "//modules/common/canbus/can_client:can_client",
        "//modules/msgs/canbus/proto:canbus_proto",
        "//modules/common",
    ],
)

cc_binary(
    name = "can_send",
    srcs = ["can_send.cc"],
    deps = [
        "//external:gflags",
        "//modules/common/canbus/can_client:can_client_factory",
        "//modules/common/canbus/can_client:can_client",
        "//modules/msgs/canbus/proto:canbus_proto",
        "//modules/common",
    ],
)

cpplint()
