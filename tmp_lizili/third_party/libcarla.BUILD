package(default_visibility = ["//visibility:public"])

cc_library(
  name = "libcarla",
  srcs = glob([
    "source/carla/**/*.cpp",
    "source/moodycamel/*.cpp"
  ]),
  hdrs = glob([
    "source/carla/**/*.h",
    "source/carla/**/*.hpp",
    "source/moodycamel/*.h",
    "source/moodycamel/*.hpp",
  ]),
  includes = [
	  "./source"
  ],
  linkopts = [
    "-lboost_system",
    "-lboost_filesystem",
  ],
  deps = [
    "@rpclib",
  ]
)
