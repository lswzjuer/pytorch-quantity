load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "check",
    srcs = ["check.cc"],
    deps = [
        "//modules/msgs/dreamview/proto:hw_monitor_result_proto",
        "@canlib",
        "@com_google_protobuf//:protobuf",
    ],
)

cpplint()
