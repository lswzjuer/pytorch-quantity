# -*- python -*-

# From https://bazel.build/versions/master/docs/be/c-cpp.html#cc_library.srcs
_SOURCE_EXTENSIONS = [source_ext for source_ext in """
.c
.cc
.cpp
.cxx
.c++
.C
.h
.hh
.hpp
.hxx
.inc
""".split("\n") if len(source_ext)]

# The cpplint.py command-line argument so it doesn't skip our files!
_LINT_ARGS = [
    "--quiet",
    "--checks=-*,google-*",
    "--header-filter=((?!third_party).)*",
]

def _extract_labels(srcs):
    """Convert a srcs= or hdrs= value to its set of labels."""

    # Tuples are already labels.
    if type(srcs) == type(()):
        return list(srcs)
    return []

def _is_source_label(label):
    for extension in _SOURCE_EXTENSIONS:
        if label.endswith(extension):
            return True
    return False

def _add_linter_rules(source_labels, source_filenames, name, deps_labels, data = None):
    # Common attributes for all of our py_test invocations.
    COMPILE_FLAGS = ""
    data = (data or [])
    size = "medium"
    tags = ["cpplint"]

    clangtidy_cfg = ["//:compile_flags.txt"] + native.glob(["compile_flags.txt"]) + ["//:.clang-tidy"] + native.glob([".clang-tidy"])
    native.sh_test(
        name = name + "_cpplint",
        srcs = ["//tools:clangtidy"],
        data = data + clangtidy_cfg + source_labels + deps_labels,
        args = _LINT_ARGS + source_filenames,
        size = size,
        tags = tags,
    )

def cpplint(data = None, extra_srcs = None):
    """For every rule in the BUILD file so far, adds a test rule that runs
    cpplint over the C++ sources listed in that rule.  Thus, BUILD file authors
    should call this function at the *end* of every C++-related BUILD file.
    By default, only the CPPLINT.cfg from the project root and the current
    directory are used.  Additional configs can be passed in as data labels.
    Sources that are not discoverable through the "sources so far" heuristic can
    be passed in as extra_srcs=[].
    """

    # Iterate over all rules.
    for rule in native.existing_rules().values():
        # Extract the list of C++ source code labels and convert to filenames.
        candidate_labels = _extract_labels(rule.get("srcs", ()))

        source_labels = [
            label
            for label in candidate_labels
            if _is_source_label(label)
        ]
        deps_labels = _extract_labels(rule.get("deps", ()))
        source_filenames = ["$(location %s)" % x for x in source_labels]

        # Run the cpplint checker as a unit test.
        if len(source_filenames) > 0:
            _add_linter_rules(source_labels, source_filenames, rule["name"], deps_labels, data)

    # Lint all of the extra_srcs separately in a single rule.
    if extra_srcs:
        source_labels = extra_srcs
        source_filenames = ["$(location %s)" % x for x in source_labels]
        _add_linter_rules(
            source_labels,
            source_filenames,
            "extra_srcs_cpplint",
            deps_labels,
            data,
        )
