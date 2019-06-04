#! /usr/bin/env python
"""
Stat disengagements and auto/manual driving mileage.
Usage:
    ./stat_mileage.py [bag|bag_dir](type:str) [whether log or not](type:bool, optional)
"""

import collections
import math
import os
import sys
import rosbag
import time
import signal

from modules.msgs.canbus.proto import chassis_pb2
from modules.msgs.canbus.proto.chassis_pb2 import Chassis
from modules.msgs.localization.proto import localization_pb2


kChassisTopic = '/roadstar/canbus/chassis'
kLocalizationTopic = '/roadstar/localization'
kTopics = [kChassisTopic, kLocalizationTopic]


class MileageCalculator(object):
    """Calculate mileage."""

    def __init__(self, conf):
        """Init."""
        self.auto_mileage = 0.0
        self.manual_mileage = 0.0
        self.disengagements = 0
        self.tot_auto_mileage = 0.0
        self.tot_manual_mileage = 0.0
        self.tot_disengagements = 0
        self.conf = conf
        self.bag_file = None

    def update(self):
        self.tot_disengagements += self.disengagements 
        self.tot_auto_mileage += self.auto_mileage
        self.tot_manual_mileage += self.manual_mileage
        if self.conf['log'] == True and self.disengagements > 0:
            self.dump_file(self.conf['log_file'])
            

    def dump_file(self, log_file):
        log_file.write(self.bag_file + ' ' + str(self.disengagements) + '\n')

    def print_log(self):
        print 'bag file: %s' % self.bag_file
        print 'Disengagements: %d' % self.disengagements
        print 'Auto mileage:   %.3f km' % (
            self.auto_mileage)
        print 'Manual mileage: %.3f km' % (
            self.manual_mileage)

        print 'Total Disengagements: %d' % self.tot_disengagements
        print 'Total Auto mileage:   %.3f km' % (
            self.tot_auto_mileage)
        print 'Total Manual mileage: %.3f km' % (
            self.tot_manual_mileage)

    def calculate(self, bag_file):
        self.auto_mileage = 0.0
        self.manual_mileage = 0.0
        self.disengagements = 0
        self.bag_file = bag_file
        last_loc = None
        last_mode = 'Unknown'
        mileage = collections.defaultdict(lambda: 0.0)
        chassis = chassis_pb2.Chassis()
        localization = localization_pb2.Localization()
        try:
            for topic, msg, t in rosbag.Bag(bag_file).read_messages(topics=kTopics):
                if topic == kChassisTopic:
                    # Mode changed
                    chassis = msg
                    if last_mode != chassis.driving_mode:
                        if (last_mode == Chassis.COMPLETE_AUTO_DRIVE and
                                chassis.driving_mode == Chassis.EMERGENCY_MODE):
                            self.disengagements += 1
                        last_mode = chassis.driving_mode
                        # Reset start position.
                        last_loc = None
                elif topic == kLocalizationTopic:
                    cur_loc = msg 
                    if last_loc:
                        # Accumulate mileage, from xyz-distance to miles.
                        mileage[last_mode] += 0.001 * math.sqrt(
                            (cur_loc.utm_x - last_loc.utm_x) ** 2 +
                            (cur_loc.utm_y - last_loc.utm_y) ** 2 +
                            (cur_loc.utm_z - last_loc.utm_z) ** 2)
                    last_loc = cur_loc
        except SystemExit:
            sys.exit()
        except:
            print 'bad input:', bag_file 
            return ;
        self.auto_mileage += mileage[Chassis.COMPLETE_AUTO_DRIVE]
        self.manual_mileage += (mileage[Chassis.COMPLETE_MANUAL] +
                                mileage[Chassis.EMERGENCY_MODE])

def create_baglist(bag_dir):
    bag_list = []
    for fpathe, dirs, fs in os.walk(bag_dir):
        for f in fs:
            item =  os.path.join(fpathe,f)
            if item.split('.')[-1]=='bag':
                bag_list.append(os.path.join(item))

    bag_list.sort()
    return bag_list


def main():
    def exit(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)
    path = sys.argv[1]
    log = False
    log_file = None
    if len(sys.argv) > 2:
        log = bool(sys.argv[2])
        if log == True:
            file_name =time.strftime(
                    '%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + '.txt'
            log_file = open(file_name, 'w')

    conf = {'log': log, 'log_file': log_file}

    mc = MileageCalculator(conf)

    if os.path.isfile(path):
        mc.calculate(path)
        mc.update()
    else:
        bag_list = create_baglist(path)
        for bagdir in bag_list:
            mc.calculate(bagdir)
            mc.update()
            mc.print_log()
    mc.print_log()
    if log == True:
        log_file.close()


if __name__ == '__main__':
    main()
