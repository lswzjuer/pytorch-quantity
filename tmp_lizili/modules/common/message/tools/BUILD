load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "record_bag",
  srcs = [
      "record_bag.cc",
  ],
  deps = [
      ":bag_tool",
      ":message_bag",
      "//external:gflags",
      "//modules/common:log",
      "//modules/common/adapters/proto:adapter_config_proto",
      "//modules/common/message:message_service",
      "//modules/common/message/tools/proto:message_bag_proto",
  ],
)

cc_binary(
  name = "play_bag",
  srcs = [
      "play_bag.cc",
  ],
  deps = [
      ":bag_tool",
      "//external:gflags",
      "//modules/common:log",
      "//modules/common/adapters/proto:adapter_config_proto",
      "//modules/common/message:message_service",
      "//modules/common/message/tools/proto:message_bag_proto",
  ],
)

cc_library(
  name = "message_bag",
  srcs = [
      "message_bag.cc",
  ],
  hdrs = [
      "message_bag.h",
  ],
  deps = [
      "//modules/common:log",
      "//modules/common/adapters/proto:adapter_config_proto",
      "//modules/common/message/tools/proto:message_bag_proto",
  ],
)

cc_test(
  name = "message_bag.test",
  size = "small",
  srcs = [
      "message_bag_test.cc",
  ],
  deps = [
      ":message_bag",
      "//modules/common:log",
      "//modules/common/message/tools/proto:message_bag_proto",
      "@gtest//:main",
  ],
)

cc_library(
  name = "bag_tool",
  srcs = [
      "bag_tool.cc",
  ],
  hdrs = [
      "bag_tool.h",
  ],
  deps = [
      ":utils",
      ":message_bag",
      "//modules/common:log",
      "//modules/common:macro",
      "//modules/common/adapters/proto:adapter_config_proto",
      "//modules/common/message/tools/proto:message_bag_proto",
  ],
)

cc_library(
  name = "utils",
  srcs = [
      "utils.cc",
  ],
  hdrs = [
      "utils.h",
  ],
  deps = [
      "//modules/common:log",
      "//modules/common:macro",
      "//modules/common/adapters/proto:adapter_config_proto",
  ],
)


cc_binary(
  name = "diagnose",
  srcs = [
      "diagnose.cc",
  ],
  deps = [
      "//modules/common:log",
      "//modules/common/util",
      "//modules/common/adapters/proto:adapter_config_proto",
      "//modules/common/message:utils",
      "//modules/common/message/proto:message_header_proto",
      "//modules/common/message/proto:diagnose_proto",
      "//modules/common/message/tools/proto:message_bag_proto",
      "//modules/common/module_conf:module_util",
      "//modules/msgs/module_conf/proto:module_conf_proto",
      "//modules/common/message:message_service",
  ],
)

cc_binary(
  name = "echo_message",
  srcs = [
      "echo_message.cc",
  ],
  deps = [
      ":utils",
      "//external:gflags",
      "//modules/common:log",
      "//modules/common/util",
      "//modules/common/adapters:adapter_manager",
      "//modules/common/message:message_service",
      "//modules/common/adapters/proto:adapter_config_proto",
      "//modules/common/message/proto:message_header_proto",
      "//modules/common/module_conf:module_util",
      "//modules/msgs/module_conf/proto:module_conf_proto",
  ],
)
