#!/usr/bin/python

import json
import sys
import os


def check_report(json_file):
    content = open(json_file, 'r')
    json_content = json.load(content)

    if "replay_diff_info" in json_content:
        if json_content["replay_diff_info"]["max_distance"] > 50:
            return 1

    fail_log = json_content["fail_log"]
    if fail_log == None:
        return 0    # means OK
    else:
        fail_cnt = {"cross line when enter or exit section": 0, "swing": 0, "slight_off_road": 0}
        for i in range(len(fail_log)):
            fail_type = fail_log[i]["type"]
            
            
            if fail_type == "off_road":
                off_rate = fail_log[i]["deviate_rate"]
                if off_rate <= 0.25:
                    fail_cnt["slight_off_road"] += 1
                    continue 
                
            if fail_type in fail_cnt:
                fail_cnt[fail_log[i]["type"]] += 1
            else:
                fail_cnt[fail_type] = 1
                    
        fail_type_num = 0
        for i in fail_cnt:
            fail_type_num += 1

        if fail_cnt["cross line when enter or exit section"] < 20 \
                and fail_cnt["swing"] < 15 \
                and fail_cnt["slight_off_road"] < 10 \
                and fail_type_num <= 3:
            return 0    # means OK
        else:
            return 1    # means fail


def main():
    in_file = sys.argv[1]

    if not os.path.isfile(in_file):
        print "1"   # means fail
    else:
        ret = check_report(json_file=in_file)
        print ret


if __name__ == "__main__":
    main()
