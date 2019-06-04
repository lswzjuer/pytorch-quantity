package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mxnet",
    srcs = [
        "lib/libmxnet.so",
    ],
    hdrs = glob([
      "include/**/*.h",
      "include/**/*.hpp",
    ]),
    include_prefix = "mxnet",
    includes = [
        "include",
    ],
)
