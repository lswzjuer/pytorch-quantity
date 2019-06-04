package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorflow",
    srcs = [
        "lib/libtensorflow_cc.so",
    ],
    hdrs = glob([
        "include/**/*.h",
    ]),
    include_prefix = "tensorflow",
    includes = [
        "include",
        "include/tensorflow/contrib/makefile/downloads",
        "include/tensorflow/contrib/makefile/downloads/fft2d",
        "include/tensorflow/contrib/makefile/downloads/absl",
        "include/tensorflow/contrib/makefile/downloads/gemmlowp",
        "include/tensorflow/contrib/makefile/downloads/nsync/public",
    ],
    deps = [
        "@eigen//:eigen",
    ]
)
