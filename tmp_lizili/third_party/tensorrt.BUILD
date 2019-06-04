package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "tensorrt",
    srcs = glob([
		    "lib/lib*.so"
		]),
    hdrs = glob([
        "include/*.h",
    ]),
    includes = ["include"],
	  include_prefix = "tensorrt",
)
