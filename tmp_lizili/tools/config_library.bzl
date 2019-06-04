load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

proto_srcs = FileType([".proto"])

def config_library(
        name,
        messages,
        protos,
        **kwargs):
    includes = ""
    for proto in protos:
        includes += """#include \\\"{}\\\"
      """.format("$(location {}).pb.h".format(proto))
    content = """#include \\\"modules/common/config/register.h\\\"
{}
""".format(includes)
    for message in messages:
        content += """REGISTER_CONFIG("::roadstar::{}");
""".format(message)
    native.genrule(
        name = name + "_config",
        outs = [name + "_config.cc"],
        cmd = "echo -e \"" + content + "\" | sed \"s/.proto.pb.h\\\"$$/.pb.h\\\"/g\"> $@",
        srcs = protos,
    )
    cc_proto_library(
        protos = protos,
        name = name,
        srcs = [name + "_config"],
        deps = ["//modules/common/config:register"],
        with_grpc = False,
        **kwargs
    )
