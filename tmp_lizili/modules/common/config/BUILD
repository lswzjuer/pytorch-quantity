load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "config",
    hdrs = [
        "config.h",
    ],
    deps = [
        "//modules/common:log",
    ],
)

cc_library(
    name = "config_manager",
    srcs = [
        "config_manager.cc",
    ],
    hdrs = [
        "config_manager.h",
    ],
    linkopts = [
        "-lstdc++fs",
    ],
    deps = [
        ":config",
        "//modules/common/util",
    ],
)

cc_library(
    name = "register",
    hdrs = [
        "register.h",
    ],
    deps = [
        ":config_manager",
    ],
)
