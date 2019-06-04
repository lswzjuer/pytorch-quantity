package(default_visibility = ["//visibility:public"])

cc_library(
    name = "common",
    srcs = glob([
      "lib/libopencv*.so",
    ]),
    hdrs = glob([
      "include/**/*.h",
      "include/**/*.hpp",
    ]),
    includes = ["include"],
    include_prefix = "opencv3",
)
