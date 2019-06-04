#!/usr/bin/python

import json
import sys
import os

label = ['0.0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']


def aggregate_deviate(dict, value):
    if value <= 0.25:
        dict[label[0]] += 1
    elif value <= 0.5:
        dict[label[1]] += 1
    elif value <= 0.75:
        dict[label[2]] += 1
    else:
        dict[label[3]] += 1


def print_json_outline(json_file):
    content = open(json_file, 'r')
    json_content = json.load(content)
    fail_log = json_content["fail_log"]
    if fail_log == None:
        fail_log = json.loads("{}")

    fail_cnt = {}
    cross_line_when_enter_or_exit_section_deviate = {
        label[0]: 0, label[1]: 0, label[2]: 0, label[3]: 0}
    off_road_deviate = {label[0]: 0, label[1]: 0, label[2]: 0, label[3]: 0}
    cross_line_deviate = {label[0]: 0, label[1]: 0, label[2]: 0, label[3]: 0}

    cross_line_when_enter_or_exit_section_legal = 0
    cross_line_when_enter_or_exit_section_illegal = 0
    for i in range(len(fail_log)):
        fail_type = fail_log[i]["type"]
        if fail_type in fail_cnt:
            fail_cnt[fail_type] += 1
        else:
            fail_cnt[fail_type] = 1
        if fail_type == 'cross line when enter or exit section':
            if fail_log[i]["legal_cross"] == "True":
                cross_line_when_enter_or_exit_section_legal += 1
            else:
                cross_line_when_enter_or_exit_section_illegal += 1
            aggregate_deviate(
                cross_line_when_enter_or_exit_section_deviate, fail_log[i]['deviate_rate'])
        elif fail_type == 'cross line':
            aggregate_deviate(cross_line_deviate, fail_log[i]['deviate_rate'])
        elif fail_type == 'off_road':
            aggregate_deviate(off_road_deviate, fail_log[i]['deviate_rate'])

    print("------ FAIL LOG ------")
    if fail_cnt == {}:
        print("no fail log")
    for fail_type in fail_cnt:
        print "%s\t%d" % (fail_type, fail_cnt[fail_type])
    if "cross line when enter or exit section" in fail_cnt:
        print('** legallity info **')
        print('legally cross_line_when_enter_or_exit_section: %d' %
              cross_line_when_enter_or_exit_section_legal)
        print('illegally cross_line_when_enter_or_exit_section: %d' %
              cross_line_when_enter_or_exit_section_illegal)
    if "off_road" in fail_cnt or "cross line" in fail_cnt or "cross line when enter or exit section" in fail_cnt:
        print('** deviate info **')
    if "off_road" in fail_cnt:
        print("off_road_deviate")
        for i in range(len(label)):
            print "%s:\t%d" % (label[i], off_road_deviate[label[i]])
    if "cross line" in fail_cnt:
        print("cross_line_deviate")
        for i in range(len(label)):
            print "%s:\t%d" % (label[i], cross_line_deviate[label[i]])
    if "cross line when enter or exit section" in fail_cnt:
        print("cross_line_when_enter_or_exit_section_deviate")
        for i in range(len(label)):
            print "%s:\t%d" % (label[i], cross_line_when_enter_or_exit_section_deviate[label[i]])

    print("------ WARN LOG ------")
    if "warn_log" not in json_content or json_content["warn_log"] == None:
        print("no warn log")
    else:
        warn_log = json_content["warn_log"]
        warn_cnt = {}
        for i in range(len(warn_log)):
            warn_type = warn_log[i]["type"]
            if warn_type in warn_cnt:
                warn_cnt[warn_type] += 1
            else:
                warn_cnt[warn_type] = 1
        for warn_type in warn_cnt:
            print "%s\t%d" % (warn_type, warn_cnt[warn_type])

    if "driving_distance" in json_content and "evaluation_duration" in json_content:
        print("----- OTHER INFO -----")
        print("driving distance(m): %f" % json_content["driving_distance"])
        print("evaluation duration(s): %f" % json_content["evaluation_duration"])

    if "replay_diff_info" in json_content:
        print ("----- REPLAY INFO -----")
        print "replay max distance: %fm" % (json_content["replay_diff_info"]["max_distance"])


def main():
    in_file = sys.argv[1]

    if not os.path.isfile(in_file):
        print "%s doesn't exist." % in_file
        return
    print_json_outline(json_file=in_file)


if __name__ == "__main__":
    main()
