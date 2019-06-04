#!/usr/bin/env python


"""
Data Collector
"""
import sys
import rospy
from std_msgs.msg import String
from modules.localization.proto import localization_pb2
from modules.canbus.proto import chassis_pb2
from modules.control.proto import control_command_pb2
import time
import os


class DataCollector(object):
    """
    DataCollector Class
    """

    def __init__(self, file):
        self.proc = [line.rstrip('\n') for line in open(file)]
        self.index = 0
        outfile = file + '_recorded.csv'
        i = 0
        outfile = file + str(i) + '_recorded.csv'
        while os.path.exists(outfile):
            i += 1
            outfile = file + str(i) + '_recorded.csv'

        self.file = open(outfile, 'w')
        self.file.write(
            "time,io,ctlmode,ctlbrake,ctlthrottle,ctlgear_location,vehicle_speed,"
            +
            "engine_rpm,driving_mode,throttle_percentage,brake_percentage,gear_location, imu\n"
        )

        self.sequence_num = 0
        self.control_pub = rospy.Publisher(
            '/roadstar/control', control_command_pb2.ControlCommand, queue_size=1)
        rospy.sleep(0.3)
        self.controlcommand = control_command_pb2.ControlCommand()

        # Send First Reset Message
        print "Send Reset Command"
        self.controlcommand.header.module_name = "control"
        self.controlcommand.header.sequence_num = self.sequence_num
        self.sequence_num = self.sequence_num + 1
        self.controlcommand.header.timestamp_sec = rospy.get_time()
        self.controlcommand.pad_msg.action = 2
        self.control_pub.publish(self.controlcommand)

        rospy.sleep(0.3)
        # Set Default Message
        print "Send Default Command"
        self.controlcommand.pad_msg.action = 1
        self.controlcommand.throttle = 0
        self.controlcommand.brake = 0
        self.controlcommand.steering_rate = 100
        self.controlcommand.steering_target = 0
        self.controlcommand.gear_location = chassis_pb2.Chassis.GEAR_NEUTRAL

        self.printedcondition = False
        self.runtimer = False
        self.canmsg_received = False
        self.localization_received = False

    def callback_localization(self, data):
        """
        New Localization
        """
        self.acceleration = data.pose.linear_acceleration_vrf.y
        self.localization_received = True

    def callback_canbus(self, data):
        """
        New CANBUS
        """
        if not self.localization_received:
            print "No Localization Message Yet"
            return
        timenow = data.header.timestamp_sec
        self.vehicle_speed = data.speed_mps
        self.engine_rpm = data.engine_rpm
        self.throttle_percentage = data.throttle_percentage
        self.brake_percentage = data.brake_percentage
        self.gear_location = data.gear_location
        self.driving_mode = data.driving_mode

        self.write_file(timenow, 0)
        self.canmsg_received = True

    def publish_control(self):
        """
        New Control Command
        """
        if not self.canmsg_received:
            print "No CAN Message Yet"
            return

        self.controlcommand.header.sequence_num = self.sequence_num
        self.sequence_num = self.sequence_num + 1

        while self.index < len(self.proc):
            commandtype = self.proc[self.index][0]
            proc = self.proc[self.index][2:].lstrip()
            if commandtype == 'a':
                command = 'self.controlcommand.' + proc
                exec (command)
                self.index = self.index + 1
                self.printedcondition = False
                print proc
            elif commandtype == 'c':
                condition = 'self.' + proc
                if eval(condition):
                    self.index = self.index + 1
                    self.printedcondition = False
                    print proc
                else:
                    if not self.printedcondition:
                        print "Waiting for condition: ", proc
                        self.printedcondition = True
                    break
            elif commandtype == 't':
                delaytime = float(proc)
                if not self.runtimer:
                    self.starttime = rospy.get_time()
                    self.runtimer = True
                    print "Waiting for time: ", delaytime
                    break
                elif rospy.get_time() > (self.starttime + delaytime):
                    self.index = self.index + 1
                    self.runtimer = False
                    print "Delayed for: ", delaytime
                else:
                    break
            else:
                print "Invalid Command, What are you doing?"
                print "Exiting"
                self.file.close()
                rospy.signal_shutdown("Shutting down")

        self.controlcommand.header.timestamp_sec = rospy.get_time()
        #self.control_pub.publish(self.controlcommand.SerializeToString())
        self.control_pub.publish(self.controlcommand)
        self.write_file(self.controlcommand.header.timestamp_sec, 1)
        if self.index >= len(self.proc):
            print "Reached end of commands, shutting down"
            self.file.close()
            rospy.signal_shutdown("Shutting down")

    def write_file(self, time, io):
        """
        Write Message to File
        """
        self.file.write(
            "%.4f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %
            (time, io, 1, self.controlcommand.brake, self.controlcommand.throttle,
             self.controlcommand.gear_location, self.vehicle_speed, self.engine_rpm,
             self.driving_mode, self.throttle_percentage, self.brake_percentage,
             self.gear_location, self.acceleration))


def main():
    """
    Main function
    """
    if len(sys.argv) <= 1:
        print "Require Command Script"
        return
    elif len(sys.argv) > 2:
        print "Too many inputs"
        return
    file = sys.argv[1]
    rospy.init_node('data_collector', anonymous=True)

    data_collector = DataCollector(file)
    localizationsub = rospy.Subscriber('/roadstar/localization/pose',
                                       localization_pb2.LocalizationEstimate,
                                       data_collector.callback_localization)
    canbussub = rospy.Subscriber('/roadstar/canbus/chassis', chassis_pb2.Chassis,
                                 data_collector.callback_canbus)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        data_collector.publish_control()
        rate.sleep()


if __name__ == '__main__':
    main()
