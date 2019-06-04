#!/usr/bin/python

import json
import sys
import os

def get_json_value(json_file, keys):
    if not os.path.isfile(json_file):
        print "1" 
        return
    content = open(json_file, 'r')
    json_content = json.load(content)
    res=[]
    for key in keys:
        res.append(str(json_content[key]))
    content.close()
    delimiter = "-"
    print delimiter.join(res)


def main():
    in_file = sys.argv[1]
    key_list = sys.argv[2:]
    get_json_value(json_file=in_file, keys=key_list)


if __name__ == "__main__":
    main()
