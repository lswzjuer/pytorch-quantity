load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "canvas_transform",
  hdrs = [
    "canvas_transform.h"
  ],
  deps = [
    "//modules/common/visualization/common:canvas_view",
    "//modules/common/coordinate:coord_trans",
    "@opencv3//:common",
    "@eigen",
  ]
)

cc_library(
  name = "geometry_drawing",
  srcs = [
    "geometry_drawing.cc",
  ],
  hdrs = [
    "geometry_drawing.h",
  ],
  deps = [
    "//modules/common/util:colormap",
    "//modules/common/proto:common_proto",  # geometry
    "@eigen",
    ":canvas_transform",
  ]
)

cc_library(
  name = "layer_renderer",
  hdrs = [
    "layer_renderer.h"
  ],
)

cc_library(
  name = "hdmap_renderer",
  srcs = [
    "hdmap_renderer.cc",
  ],
  hdrs = [
    "hdmap_renderer.h"
  ],
  deps = [
    "//modules/msgs/hdmap/proto:hdmap_proto",
    "//modules/common/util:colormap",
    ":geometry_drawing",
    ":canvas_transform",
    ":layer_renderer",
    "//modules/common/visualization/common/proto:hdmap_render_param",
  ]
)

cc_library(
  name = "fusion_map_renderer",
  srcs = [
    "fusion_map_renderer.cc"
  ],
  hdrs = [
    "fusion_map_renderer.h",
  ],
  deps = [
    "//modules/msgs/perception/proto:perception_proto",
    "//modules/common/util:colormap",
    ":geometry_drawing",
    ":canvas_transform",
    ":layer_renderer",
    "//modules/common/visualization/common/proto:fusion_map_render_param",
  ]
)

cc_library(
  name = "ego_renderer",
  srcs = [
    "ego_renderer.cc"
  ],
  hdrs = [
    "ego_renderer.h",
  ],
  deps = [
    "//modules/msgs/localization/proto:localization_proto",
    "//modules/common/util:colormap",
    ":geometry_drawing",
    ":canvas_transform",
    ":layer_renderer",
    "//modules/common/visualization/common/proto:ego_render_param",
  ]
)

cpplint()
