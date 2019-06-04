load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "label_data_transform",
    srcs = [
        "label_data_transform.cc",
    ],
    deps = [
        "//modules/common:log",
        "//modules/common/util",
        "//modules/integration_test/common/xml_param:xml_param_reader",
        "//modules/integration_test/perception/common/model",
        "//modules/integration_test/perception/obstacle/model",
        "//modules/integration_test/perception/obstacle/serialize:label_data_parser",
        "//modules/integration_test/perception/util:ros_bag_view",
        "//modules/msgs/perception/proto:perception_proto",
        "//third_party/json",
        "@ros//:ros_common",
    ],
)

cpplint()
