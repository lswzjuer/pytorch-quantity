package(default_visibility = ["//visibility:public"])

cc_library(
    name = "coord_trans",
    hdrs = [
        "coord_trans.h",
    ],
    deps = [
        "@eigen",
    ]
)

cc_library(
    name = "sensor_coordinate",
    srcs = [
        "sensor_coordinate.cc",
    ],
    hdrs = [
        "sensor_coordinate.h",
    ],
    deps = [
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/common:common_gflags",
        "//modules/common:sensor_source",
        "//modules/common/coordinate/proto:calibration_proto",
        "//modules/common/util:util",
        "//modules/common/adapters:adapter_manager",
        ":coord_trans",
        "@eigen",
    ],
)

cc_library(
    name = "camera_coordinate",
    srcs = [
        "camera_coordinate.cc",
    ],
    hdrs = [
        "camera_coordinate.h",
        "camera.h",
    ],
    deps = [
        "//modules/common:common_gflags",
        "//modules/common:sensor_source",
        "//modules/common/coordinate/proto:calibration_proto",
        "//modules/common/util:util",
        "@eigen",
        "@opencv3//:common",
    ],
)

cc_test(
    name = "sensor_coordinate_test",
    size = "small",
    srcs = [
        "sensor_coordinate_test.cc"
        ],
    data = [
        "//resources/calibration:perception_data",
    ],
    deps = [
        ":sensor_coordinate",
        "@gtest//:main",
        ],
)

cc_test(
    name = "camera_coordinate_test",
    size = "small",
    srcs = [
        "camera_coordinate_test.cc"
        ],
    data = [
        "//resources/calibration:perception_data",
    ],
    deps = [
        ":camera_coordinate",
        "@gtest//:main",
        "@opencv3//:common",
        "//modules/common/adapters:adapter_manager",
        ],
)
