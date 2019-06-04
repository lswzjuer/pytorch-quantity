package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "common",
    srcs = [
        "lib/libcaffe.so",
    ],
    hdrs = glob([
      "include/**/*.h",
      "include/**/*.hpp",
    ]),
    includes = ["include"],
    include_prefix = "caffe",
    linkopts = [
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu/",
        "-L/usr/lib/x86_64-linux-gnu/hdf5/serial",
        "-L/usr/local/cuda/lib64",
        "-lboost_system",
        "-lboost_thread",
        "-lboost_filesystem",
        "-lpthread",
        "-lcblas",
        "-lcurand",
        "-lcudart",
        "-lcublas",
        "-lcudnn",
        "-lz",
        "-ldl",
        "-lm",
    ] + select({
			":g3logger": [
				"-L/usr/local/lib",
        "-lglog",
      ],
			":googlelogger": [],
		}),
    deps = [
      "@cuda",
    ]
)

config_setting(
  name = "g3logger",
  values = {"define": "logger=g3log"},
)

config_setting(
  name = "googlelogger",
  values = {"define": "logger=glog"},
)

