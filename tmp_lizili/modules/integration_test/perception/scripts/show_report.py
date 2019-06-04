#!/usr/bin/python

import json
import sys
import os

def print_json_value(json_file, keys):
    if not os.path.isfile(json_file):
        print "%s doesn't exist." % json_file
        return
    content = open(json_file, 'r')
    json_content = json.load(content)
    for key in keys:
        print "%s = %s" % (key, json_content[key])
    content.close()


def main():
    in_file = sys.argv[1]
    key_list = sys.argv[2:]
    print_json_value(json_file=in_file, keys=key_list)


if __name__ == "__main__":
    main()
