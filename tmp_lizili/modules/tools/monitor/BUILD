load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "log_monitor",
    srcs = ["log_monitor.cc"],
    deps = [
        "//modules/common:log",
        "//modules/common/time",
        "//modules/common/util:concurrent_queue",
    ],
)

cpplint()
