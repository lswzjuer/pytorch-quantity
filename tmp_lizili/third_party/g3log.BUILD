cc_library(
    name = "base_g3log",
    srcs = glob([
        "src/*.ipp",
        "src/*.cpp",
    ]),
    hdrs = glob([
        "src/g3log/*.hpp",
    ]),
    linkopts = [
        "-lstdc++fs",
    ],
    defines = [
        "G3_DYNAMIC_LOGGING",
    ],
    includes = [
        "src/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//external:gflags",
    ],
)

cc_library(
    name = "g3log",
    srcs = [
        "src/g3log/sinks/custom_sink.cpp",
    ],
    hdrs = [
        "src/g3log/sinks/custom_sink.hpp",
    ],
    includes = [
        "src/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base_g3log",
        "//external:gflags",
    ],
)
