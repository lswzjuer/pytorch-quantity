load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "id",
    hdrs = [
        "id.h",
    ],
)

cc_library(
    name = "map_object",
    srcs = [
        "connection_object.cc",
        "road_edge_object.cc",
        "section_object.cc",
    ],
    hdrs = [
        "connection_object.h",
        "map_object.h",
        "road_edge_object.h",
        "section_object.h",
    ],
    deps = [
        ":id",
        "//modules/common/math:geometry",
        "//modules/msgs/hdmap/proto:hdmap_proto",
    ],
)

cc_library(
    name = "map_unit_hash",
    hdrs = [
        "map_unit_hash.h",
    ],
    deps = [
        "//modules/msgs/hdmap/proto:hdmap_common_proto",
    ],
)

cc_library(
    name = "map",
    hdrs = [
        "map.h",
    ],
    deps = [
        ":map_object",
        ":map_unit_hash",
        "//modules/common/math",
    ],
)

cpplint()
