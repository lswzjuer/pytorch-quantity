load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "qp_solver_gflags",
    srcs = [
        "qp_solver_gflags.cc",
    ],
    hdrs = [
        "qp_solver_gflags.h",
    ],
    deps = [
        "//external:gflags",
    ],
)

cc_library(
    name = "qp_solver",
    srcs = [
        "qp_solver.cc",
    ],
    hdrs = [
        "qp_solver.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
)

cc_library(
    name = "active_set_qp_solver",
    srcs = [
        "active_set_qp_solver.cc",
    ],
    hdrs = [
        "active_set_qp_solver.h",
    ],
    deps = [
        ":qp_solver",
        "//modules/common:log",
        "//modules/common/math/qp_solver:qp_solver_gflags",
        "@eigen//:eigen",
        "@qp_oases//:qp_oases",
    ],
)

cc_library(
    name = "gpad_qp_solver",
    srcs = [
        "gpad_qp_solver.cc",
    ],
    hdrs = [
        "gpad_qp_solver.h",
    ],
    deps = [
        ":qp_solver",
        "//modules/common:log",
        "//modules/common/time",
        "@eigen",
    ],
)

cc_library(
    name = "admm_qp_solver",
    srcs = [
        "admm_qp_solver.cc",
    ],
    hdrs = [
        "admm_qp_solver.h",
    ],
    deps = [
        ":qp_solver",
        "//modules/common:log",
        "//modules/common/time",
        "@eigen",
    ],
)

cc_test(
    name = "active_set_qp_solver_test",
    size = "small",
    srcs = [
        "active_set_qp_solver_test.cc",
    ],
    deps = [
        ":active_set_qp_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "GPADQpSolverTest",
    size = "small",
    srcs = [
        "gpad_qp_solver_test.cc",
    ],
    deps = [
        ":gpad_qp_solver",
        "@gtest//:main",
    ],
)

cc_test(
    name = "ADMMQpSolverTest",
    size = "small",
    srcs = [
        "admm_qp_solver_test.cc",
    ],
    deps = [
        ":admm_qp_solver",
        "@gtest//:main",
    ],
)

cpplint()
