load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "integration_test_gflags",
    srcs = [
        "integration_test_gflags.cc",
    ],
    hdrs = [
        "integration_test_gflags.h",
    ],
    deps = [
        "//external:gflags",
    ],
)

cpplint()