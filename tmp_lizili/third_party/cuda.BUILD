package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "cuda",
    srcs = glob([
		    "lib64/lib*.so"
		]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
	  include_prefix = "cuda",
)
