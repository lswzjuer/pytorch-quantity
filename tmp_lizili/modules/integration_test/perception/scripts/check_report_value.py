#!/usr/bin/python

import sys
import os
import json

def get_json(json_file):
    content = open(json_file, 'r')
    json_content = json.load(content)
    content.close()
    return json_content


def check_report(json_file):
    json_content = get_json(json_file)
    precision_average = json_content["precision_average"]
    recall_average = json_content["recall_average"]
    velocity_precision_p50 = json_content["velocity_precision_p50"]
    velocity_precision_p95 = json_content["velocity_precision_p95"]
    if precision_average > 0.94 and recall_average > 0.90 \
            and velocity_precision_p50 > 0.93 and velocity_precision_p95 > 0.84:
        return 0   #means OK
    else:
        return 1   #means fail


def main():
    in_file = sys.argv[1]
    if os.path.isfile(in_file):
        ret = check_report(in_file)
        print ret
    else:
        print "1"


if __name__ == "__main__":
    main()
