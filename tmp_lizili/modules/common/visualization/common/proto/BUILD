load("//tools:cpplint.bzl", "cpplint")
load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
  name = "hdmap_render_param",
  protos = [
    "hdmap_render.proto",
  ],
  deps = [
    "//modules/common/proto:colormap_proto",
  ],
)

cc_proto_library(
  name = "fusion_map_render_param",
  protos = [
    "fusion_map_render.proto",
  ],
  deps = [
    "//modules/common/proto:colormap_proto",
  ],
)

cc_proto_library(
  name = "ego_render_param",
  protos = [
    "ego_render.proto",
  ],
  deps = [
    "//modules/common/proto:colormap_proto",
  ],
)

cpplint()
