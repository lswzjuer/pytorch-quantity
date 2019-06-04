CPP_LANG_DEPS = {

    # Grpc repo is required by multiple languages but we put it here.
    # This is the source or "base" archive for the 'grpc_repository'
    # rule, were we'll reconstruct a new repo by symlinking resources
    # from here into 'com_google_grpc'.
    "com_google_grpc_base": {
        "rule": "http_archive",
        "url": "http://git.fabu.ai/third_party/grpc/repository/archive.tar.gz?ref=v1.6.2",
        "strip_prefix": "grpc-v1.6.2-c1d9c06402a02230cd8856824b29c58a6d0b7576",
    },
    "com_google_grpc": {
        "rule": "grpc_repository",
        "base_workspace": "@com_google_grpc_base//:WORKSPACE",
    },
    "com_github_c_ares_c_ares": {
        "rule": "new_http_archive",
        "url": "http://git.fabu.ai/third_party/c-ares/repository/archive.tar.gz?ref=cares-1_12_0",
        "strip_prefix": "c-ares-cares-1_12_0-7691f773af79bf75a62d1863fd0f13ebf9dc51b1",
        "build_file_content": "",
    },
    "com_github_grpc_grpc": {
        "rule": "grpc_repository",
    },

    # Hooray! The boringssl team provides a "chromium-stable-with-bazel" branch
    # with all BUILD files ready to go.
    "boringssl": {
        "rule": "http_archive",
        "url": "http://git.fabu.ai/third_party/chromium-stable-with-bazel/raw/master/chromium-stable-with-bazel.zip",
    },

    # libssl is required for c++ grpc where it is expected in
    # //external:libssl.  This can be either boringssl or openssl.
    "libssl": {
        "rule": "bind",
        "actual": "@boringssl//boringssl-chromium-stable-with-bazel:ssl",
    },

    # C-library for zlib
    "com_github_madler_zlib": {
        "rule": "new_git_repository",
        "remote": "http://git.fabu.ai/third_party/zlib.git",
        "tag": "v1.2.11",
        "build_file": "third_party/com_github_madler_zlib.BUILD",
    },

    # grpc++ expects //external:cares
    "cares": {
        "rule": "bind",
        "actual": "@com_google_grpc//third_party/cares:ares",
    },

    # grpc++ expects //external:zlib
    "zlib": {
        "rule": "bind",
        "actual": "@com_github_madler_zlib//:zlib",
    },

    # grpc++ expects //external:nanopb
    "nanopb": {
        "rule": "bind",
        "actual": "@com_google_grpc//third_party/nanopb",
    },

    # Bind the executable cc_binary grpc plugin into
    # //external:protoc_gen_grpc_cpp.  Expects
    # //external:protobuf_compiler. TODO: is it really necessary to
    # bind it in external?
    "protoc_gen_grpc_cpp": {
        "rule": "bind",
        "actual": "@com_google_grpc//:grpc_cpp_plugin",
    },

    # GTest is for our own internal cc tests.
    "com_google_googletest": {
        "rule": "git_repository",
        "remote": "http://git.fabu.ai/third_party/googletest.git",
        "commit": "7c6353d29a147cad1c904bf2957fd4ca2befe135",
    },
}
