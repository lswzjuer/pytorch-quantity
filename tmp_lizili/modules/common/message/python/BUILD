load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "message_pyutils.so",
    srcs = [
        "message_pyutils.cc",
    ],
    copts = ["-fPIC"],
    linkshared = True,
    deps = [
        "//external:gflags",
        "//modules/common:log",
        "//modules/common/message:message_service",
        "//modules/common/message/tools:message_bag",
        "//modules/common/message/tools/proto:message_bag_proto",
    ],
)

genrule(
    name = "lib_message_pyutils_so",
    srcs = ["message_pyutils.so"],
    outs = ["lib_message_pyutils.so"],
    cmd = "cp ./$(location message_pyutils.so) \"$@\"",
)

genrule(
    name = "message_python_init",
    srcs = ["init_message_python.py"],
    outs = ["__init__.py"],
    cmd = "cp ./$(location init_message_python.py) \"$@\"",
)
