package(default_visibility = ["//visibility:public"])

cc_library(
  name = "rpclib",
  srcs = glob([
    "lib/*.cc",
    "lib/**/*.cc",
  ]),
  hdrs = glob([
    "include/rpc/*.h",
    "include/rpc/*.hpp",
    "include/rpc/**/*.h",
    "include/rpc/**/*.hpp",
  ]),
  copts = [
    "-Iinclude",
  ],
  includes = ["include"],
  include_prefix = "rpc",
  defines = [
    "RPCLIB_ASIO=clmdep_asio",
    "RPCLIB_FMT=clmdep_fmt",
    "RPCLIB_MSGPACK=clmdep_msgpack",
  ],
  deps = [
    ":rpclib_dependencies"
  ]
)

cc_library(
  name = "rpclib_dependencies",
  srcs = glob([
    "dependencies/src/*.cc",
    "dependencies/src/*.cpp",
    ]),
  hdrs = glob([
    "dependencies/include/*.h",
    "dependencies/include/*.hpp",
    "dependencies/include/**/*.h",
    "dependencies/include/**/*.hpp",
  ]),
  copts = [
    "-DASIO_SEPARATE_COMPILATION",
    "-DASIO_STANDALONE",
    "-lboost",
  ],
  linkopts = [
    "-lssl",
    "-lcrypto",
  ],
  includes = ["dependencies/include"],
)

