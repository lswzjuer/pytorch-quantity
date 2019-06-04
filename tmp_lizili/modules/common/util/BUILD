load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "util",
    srcs = [
        "file.cc",
        "string_tokenizer.cc",
        "util.cc",
    ],
    hdrs = [
        "file.h",
        "string_tokenizer.h",
        "util.h",
    ],
    deps = [
        "//modules/common:log",
        "//modules/common:macro",
        "//modules/common/math",
        "//modules/common/proto:common_proto",
        "//modules/common/proto:pnc_point_proto",
        "//modules/common/time",
        "//modules/msgs/drivers/proto:coords_calibration_proto",
        "//modules/msgs/localization/proto:localization_proto",
        "//modules/msgs/perception/proto:perception_proto",
        "@eigen",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = [
        "util_test.cc",
    ],
    deps = [
        ":util",
        "@gtest//:main",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
)

cc_library(
    name = "thread_pool",
    hdrs = ["thread_pool.h"],
)

cc_test(
    name = "thread_pool_test",
    size = "small",
    srcs = [
        "thread_pool_test.cc",
    ],
    deps = [
        "//modules/common/util:thread_pool",
        "@gtest//:main",
    ],
)

cc_library(
    name = "concurrent_queue",
    hdrs = ["concurrent_queue.h"],
)

cc_test(
    name = "concurrent_queue_test",
    size = "small",
    srcs = [
        "concurrent_queue_test.cc",
    ],
    deps = [
        "//modules/common/util:concurrent_queue",
        "@gtest//:main",
    ],
)

cc_library(
    name = "string_util",
    srcs = [
    ],
    hdrs = [
        "string_util.h",
    ],
    deps = [
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "string_util_test",
    size = "small",
    srcs = [
        "string_util_test.cc",
    ],
    deps = [
        ":string_util",
        "@com_google_protobuf//:protobuf",
        "@gtest//:main",
    ],
)

cc_library(
    name = "map_util",
    hdrs = ["map_util.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "string_tokenizer_test",
    size = "small",
    srcs = [
        "string_tokenizer_test.cc",
    ],
    deps = [
        ":util",
        "@gtest//:main",
    ],
)

cc_library(
    name = "factory",
    hdrs = [
        "factory.h",
    ],
    deps = [
        "//modules/common:log",
    ],
)

cc_test(
    name = "factory_test",
    size = "small",
    srcs = [
        "factory_test.cc",
    ],
    deps = [
        ":factory",
        "//modules/common",
        "//modules/common:log",
        "@gtest//:main",
    ],
)

cc_test(
    name = "file_test",
    size = "small",
    srcs = [
        "file_test.cc",
    ],
    tags = [
        "external",
    ],
    deps = [
        ":util",
        "//modules/common/util/testdata:simple_proto",
        "@gtest//:main",
    ],
)

cc_test(
    name = "lru_cache_test",
    size = "small",
    srcs = [
        "lru_cache_test.cc",
    ],
    deps = [
        "//modules/common/util:lru_cache",
        "@gtest//:main",
    ],
)

cc_library(
    name = "points_downsampler",
    hdrs = [
        "points_downsampler.h",
    ],
    deps = [
        "//modules/common:log",
        "//modules/common/math:geometry",
    ],
)

cc_test(
    name = "points_downsampler_test",
    size = "small",
    srcs = [
        "points_downsampler_test.cc",
    ],
    deps = [
        ":points_downsampler",
        "@gtest//:main",
    ],
)

cc_library(
    name = "singleton",
    hdrs = [
        "singleton.h",
    ],
    deps = [
    ],
)

cc_test(
    name = "singleton_test",
    size = "small",
    srcs = [
        "singleton_test.cc",
    ],
    deps = [
        ":singleton",
        "@gtest//:main",
    ],
)

cc_library(
    name = "json_util",
    srcs = [
        "json_util.cc",
    ],
    hdrs = [
        "json_util.h",
    ],
    deps = [
        "//modules/common:log",
        "//third_party/json",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "json_util_test",
    size = "small",
    srcs = [
        "json_util_test.cc",
    ],
    deps = [
        ":json_util",
        "//modules/common/proto:error_code_proto",
        "@gtest//:main",
    ],
)

cc_library(
    name = "disjoint_set",
    hdrs = [
        "disjoint_set.h",
    ],
)

cc_test(
    name = "disjoint_set_test",
    size = "small",
    srcs = [
        "disjoint_set_test.cc",
    ],
    deps = [
        ":disjoint_set",
        "//modules/common:log",
        "@gtest//:main",
    ],
)

cc_library(
    name = "http_client",
    srcs = ["http_client.cc"],
    hdrs = ["http_client.h"],
    deps = [
        "//modules/common:log",
        "//modules/common/status",
        "//third_party/json",
        "@curlpp",
    ],
)

cc_library(
    name = "hash",
    hdrs = [
        "hash.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "colormap",
    srcs = ["colormap.cc"],
    hdrs = [
        "colormap.h",
    ],
    deps = [
        ":hash",
        "//modules/common/proto:colormap_proto",
        "@opencv3//:common",
    ],
)

cc_library(
    name = "gpu_util",
    srcs = [
        "gpu_util.cc",
    ],
    hdrs = [
        "gpu_util.h",
    ],
    linkopts = [
        "-lboost_regex",
    ],
    deps = [
        ":util",
        "//modules/common:log",
    ],
)

cc_library(
    name = "recurrent_runner",
    srcs = ["recurrent_runner.cc"],
    hdrs = ["recurrent_runner.h"],
    deps = [
        "//modules/common:log",
        "//modules/common/time",
    ],
)

cc_test(
    name = "gpu_util_test",
    size = "small",
    srcs = [
        "gpu_util_test.cc",
    ],
    deps = [
        ":gpu_util",
        "@gtest//:main",
    ],
)

cc_library(
    name = "registerer",
    srcs = [
        "registerer.cc",
    ],
    hdrs = [
        "registerer.h",
    ],
    deps = [
        "//modules/common",
        "//modules/common:log",
    ],
)

cc_library(
    name = "thread",
    srcs = [
        "thread.cc",
    ],
    hdrs = [
        "thread.h",
    ],
    deps = [
        "//modules/common",
        "//modules/common:log",
    ],
)

cc_library(
    name = "type_map",
    srcs = [
    ],
    hdrs = [
        "type_map.h",
    ],
    deps = [
    ],
)

cpplint()
