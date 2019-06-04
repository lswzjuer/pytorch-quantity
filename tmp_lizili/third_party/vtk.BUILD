package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "vtk",
    srcs = glob([
		    "lib/x86_64-linux-gnu/libvtk*.so"
		]),
    hdrs = glob([
        "include/vtk-6.2/*.h",
        "include/vtk-6.2/**/*.h"
    ]),
    includes = ["include/vtk-6.2"],
	  include_prefix = "vtk"
)
