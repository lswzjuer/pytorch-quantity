package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "pylon",
    srcs = glob([
		    "lib64/lib*.so",
		]),
    hdrs = glob([
        "include/*.h",
        "include/**/*.h"
    ]),
    includes = [
        "include",
    ]
)
