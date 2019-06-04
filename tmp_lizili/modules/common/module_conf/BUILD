load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "module_util",
  srcs = [
      "module_util.cc",
  ],
  hdrs = [
      "module_util.h",
  ],
  deps = [
      "//modules/common",
      "//modules/common:log",
      "//modules/common/util",
      "//modules/msgs/module_conf/proto:module_conf_proto",
  ],
)

filegroup(
    name = "conf",
    srcs = glob([
        "conf/**",
    ]),
)
