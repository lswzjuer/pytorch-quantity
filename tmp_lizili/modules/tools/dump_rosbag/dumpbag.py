#!/usr/bin/env python


"""
This program can dump a rosbag into separate text files that contains the pb messages
"""

import rosbag
import std_msgs
import argparse
import shutil
import os
import sys

from std_msgs.msg import String


def write_to_file(file_path, topic_pb):
    """write pb message to file"""
    f = file(file_path, 'w')
    f.write(str(topic_pb))
    f.close()


def dump_bag(in_bag, out_dir, filter_topic):
    """out_bag = in_bag + routing_bag"""
    bag = rosbag.Bag(in_bag, 'r')
    seq = 0
    for topic, msg, t in bag.read_messages():
        if not filter_topic or (filter_topic and topic == filter_topic):
            message_file = topic.replace("/", "_")
            file_path = os.path.join(out_dir,
                                     str(seq) + message_file + ".pb.txt")
            write_to_file(file_path, msg)
        seq += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "A tool to dump the protobuf messages in a ros bag into text files")
    parser.add_argument(
        "in_rosbag", action="store", type=str, help="the input ros bag")
    parser.add_argument(
        "out_dir",
        action="store",
        help="the output directory for the dumped file")
    parser.add_argument(
        "--topic",
        action="store",
        help="""the topic that you want to dump. If this option is not provided,
        the tool will dump all the messages regardless of the message topic.""")
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    dump_bag(args.in_rosbag, args.out_dir, args.topic)
