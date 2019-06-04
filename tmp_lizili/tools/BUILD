package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "clangtidy",
    srcs = [
        "clangtidy.sh",
    ],
)

config_setting(
    name = "g3logger",
    values = {"define": "logger=g3log"},
)

config_setting(
    name = "googlelogger",
    values = {"define": "logger=glog"},
)
