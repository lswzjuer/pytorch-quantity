#!/usr/bin/env python


"""
Control Planning Analyzer
"""
import sys
import rospy
import rosbag
import matplotlib
from std_msgs.msg import String
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import lines
import numpy
from modules.localization.proto import localization_pb2
from modules.canbus.proto import chassis_pb2
from modules.planning.proto import planning_pb2
from modules.control.proto import control_command_pb2
import time
import tf
import math
import argparse
import threading
import tkFileDialog


class ControlInfo(object):
    """
    ControlInfo Class
    """

    def __init__(self, axarr):
        self.throttlecommand = []
        self.throttlefbk = []
        self.brakecommand = []
        self.brakefbk = []
        self.steercommand = []
        self.steerfbk = []
        self.speed = []
        self.curvature = []
        self.imuright = []
        self.imuforward = []
        self.imuup = []
        self.controltime = []
        self.planningtime = []
        self.localizationtime = []
        self.canbustime = []

        self.acceleration_lookup = []
        self.speed_lookup = []
        self.acc_open = []
        self.acc_close = []
        self.station_error = []
        self.speed_error = []

        self.heading_error = []
        self.lateral_error = []
        self.heading_error_rate = []
        self.lateral_error_rate = []

        self.target_speed = []
        self.target_curvature = []
        self.target_acceleration = []
        self.target_heading = []
        self.target_time = []

        self.driving_mode = 0
        self.mode_time = []

        self.ax = axarr

        self.planningavailable = False

        self.lock = threading.Lock()

    def callback_planning(self, entity):
        """
        New Planning Trajectory
        """
        basetime = entity.header.timestamp_sec
        numpoints = len(entity.adc_trajectory_point)
        with self.lock:
            self.pointx = numpy.zeros(numpoints)
            self.pointy = numpy.zeros(numpoints)
            self.pointspeed = numpy.zeros(numpoints)
            self.pointtime = numpy.zeros(numpoints)
            self.pointtheta = numpy.zeros(numpoints)
            self.pointcurvature = numpy.zeros(numpoints)
            self.pointacceleration = numpy.zeros(numpoints)

            for idx in range(numpoints):
                self.pointx[idx] = entity.adc_trajectory_point[idx].x
                self.pointy[idx] = entity.adc_trajectory_point[idx].y
                self.pointspeed[idx] = entity.adc_trajectory_point[idx].speed
                self.pointtheta[idx] = entity.adc_trajectory_point[idx].theta
                self.pointcurvature[idx] = entity.adc_trajectory_point[
                    idx].curvature
                self.pointacceleration[idx] = entity.adc_trajectory_point[
                    idx].acceleration_s
                self.pointtime[
                    idx] = entity.adc_trajectory_point[idx].relative_time + basetime

        if numpoints == 0:
            self.planningavailable = False
        else:
            self.planningavailable = True

    def callback_canbus(self, entity):
        """
        New Canbus
        """
        self.throttlefbk.append(entity.throttle_percentage)
        self.brakefbk.append(entity.brake_percentage)
        self.steerfbk.append(entity.steering_percentage)
        self.speed.append(entity.speed_mps)
        self.canbustime.append(entity.header.timestamp_sec)

        if entity.driving_mode == chassis_pb2.Chassis.COMPLETE_AUTO_DRIVE:
            if self.driving_mode == 0:
                self.mode_time.append(entity.header.timestamp_sec)
                self.driving_mode = 1
        elif self.driving_mode == 1:
            self.mode_time.append(entity.header.timestamp_sec)
            self.driving_mode = 0

    def callback_localization(self, entity):
        """
        New Localization
        """
        self.imuright.append(entity.pose.linear_acceleration_vrf.x)
        self.imuforward.append(entity.pose.linear_acceleration_vrf.y)
        self.imuup.append(entity.pose.linear_acceleration_vrf.z)
        self.localizationtime.append(entity.header.timestamp_sec)

    def callback_control(self, entity):
        """
        New Control Command
        """
        self.throttlecommand.append(entity.throttle)
        self.brakecommand.append(entity.brake)
        self.steercommand.append(entity.steering_target)
        self.controltime.append(entity.header.timestamp_sec)

        self.acceleration_lookup.append(
            entity.debug.simple_lon_debug.acceleration_lookup)
        self.speed_lookup.append(entity.debug.simple_lon_debug.speed_lookup)
        self.acc_open.append(
            entity.debug.simple_lon_debug.preview_acceleration_reference)
        self.acc_close.append(
            entity.debug.simple_lon_debug.acceleration_command_closeloop)
        self.station_error.append(entity.debug.simple_lon_debug.station_error)
        self.speed_error.append(entity.debug.simple_lon_debug.speed_error)

        self.curvature.append(entity.debug.simple_lat_debug.curvature)
        self.heading_error.append(entity.debug.simple_lat_debug.heading_error)
        self.lateral_error.append(entity.debug.simple_lat_debug.lateral_error)
        self.heading_error_rate.append(
            entity.debug.simple_lat_debug.heading_error_rate)
        self.lateral_error_rate.append(
            entity.debug.simple_lat_debug.lateral_error_rate)

        with self.lock:
            if self.planningavailable:
                self.target_speed.append(
                    numpy.interp(entity.header.timestamp_sec, self.pointtime,
                                 self.pointspeed))
                self.target_curvature.append(
                    numpy.interp(entity.header.timestamp_sec, self.pointtime,
                                 self.pointcurvature))
                self.target_acceleration.append(
                    numpy.interp(entity.header.timestamp_sec, self.pointtime,
                                 self.pointacceleration))
                self.target_heading.append(
                    numpy.interp(entity.header.timestamp_sec, self.pointtime,
                                 self.pointtheta))
                self.target_time.append(entity.header.timestamp_sec)

    def long(self):
        """
        
        """
        print "Showing Longitudinal"
        for loc, ax in numpy.ndenumerate(self.ax):
            ax.clear()
        self.ax[0, 0].plot(
            self.canbustime, self.throttlefbk, label='Throttle Feedback')
        self.ax[0, 0].plot(
            self.controltime, self.throttlecommand, label='Throttle Command')
        self.ax[0, 0].plot(
            self.canbustime, self.brakefbk, label='Brake Feedback')
        self.ax[0, 0].plot(
            self.controltime, self.brakecommand, label='Brake Command')
        self.ax[0, 0].legend(fontsize='medium')
        self.ax[0, 0].grid(True)
        self.ax[0, 0].set_title('Throttle Brake Info')
        self.ax[0, 0].set_xlabel('Time')

        self.ax[0, 1].plot(
            self.speed_lookup, self.acceleration_lookup, label='Table Lookup')
        self.ax[0, 1].plot(
            self.target_speed, self.target_acceleration, label='Target')
        self.ax[0, 1].legend(fontsize='medium')
        self.ax[0, 1].grid(True)
        self.ax[0, 1].set_title('Calibration Lookup')
        self.ax[0, 1].set_xlabel('Speed')
        self.ax[0, 1].set_ylabel('Acceleration')

        self.ax[1, 0].plot(self.canbustime, self.speed, label='Vehicle Speed')
        self.ax[1, 0].plot(
            self.target_time, self.target_speed, label='Target Speed')
        self.ax[1, 0].plot(
            self.target_time, self.target_acceleration, label='Target Acc')
        self.ax[1, 0].plot(
            self.localizationtime, self.imuforward, label='IMU Forward')
        self.ax[1, 0].legend(fontsize='medium')
        self.ax[1, 0].grid(True)
        self.ax[1, 0].set_title('Speed Info')
        self.ax[1, 0].set_xlabel('Time')

        self.ax[1, 1].plot(
            self.controltime, self.acceleration_lookup, label='Lookup Acc')
        self.ax[1, 1].plot(self.controltime, self.acc_open, label='Acc Open')
        self.ax[1, 1].plot(self.controltime, self.acc_close, label='Acc Close')
        self.ax[1, 1].plot(
            self.controltime, self.station_error, label='station_error')
        self.ax[1, 1].plot(
            self.controltime, self.speed_error, label='speed_error')
        self.ax[1, 1].legend(fontsize='medium')
        self.ax[1, 1].grid(True)
        self.ax[1, 1].set_title('IMU Info')
        self.ax[1, 1].set_xlabel('Time')

        if len(self.mode_time) % 2 == 1:
            self.mode_time.append(self.controltime[-1])
        for i in range(0, len(self.mode_time), 2):
            self.ax[0, 0].axvspan(
                self.mode_time[i], self.mode_time[i + 1], fc='0.1', alpha=0.1)
            self.ax[1, 0].axvspan(
                self.mode_time[i], self.mode_time[i + 1], fc='0.1', alpha=0.1)
            self.ax[1, 1].axvspan(
                self.mode_time[i], self.mode_time[i + 1], fc='0.1', alpha=0.1)
        plt.draw()

    def lat(self):
        """
        Plot everything in time domain
        """
        print "Showing Lateral"
        for loc, ax in numpy.ndenumerate(self.ax):
            ax.clear()
        self.ax[0, 0].plot(
            self.canbustime, self.steerfbk, label='Steering Feedback')
        self.ax[0, 0].plot(
            self.controltime, self.steercommand, label='Steering Command')
        self.ax[0, 0].plot(self.controltime, self.curvature, label='Curvature')
        self.ax[0, 0].legend(fontsize='medium')
        self.ax[0, 0].grid(True)
        self.ax[0, 0].set_title('Steering Info')
        self.ax[0, 0].set_xlabel('Time')
        """
        self.ax[0, 1].legend(fontsize = 'medium')
        self.ax[0, 1].grid(True)
        self.ax[0, 1].set_title('Calibration Lookup')
        self.ax[0, 1].set_xlabel('Speed')
        self.ax[0, 1].set_ylabel('Acceleration')
        """

        self.ax[1, 0].plot(
            self.controltime, self.heading_error, label='heading_error')
        self.ax[1, 0].plot(
            self.controltime, self.lateral_error, label='lateral_error')
        self.ax[1, 0].legend(fontsize='medium')
        self.ax[1, 0].grid(True)
        self.ax[1, 0].set_title('Error Info')
        self.ax[1, 0].set_xlabel('Time')

        self.ax[1, 1].plot(
            self.controltime,
            self.heading_error_rate,
            label='heading_error_rate')
        self.ax[1, 1].plot(
            self.controltime,
            self.lateral_error_rate,
            label='lateral_error_rate')
        self.ax[1, 1].legend(fontsize='medium')
        self.ax[1, 1].grid(True)
        self.ax[1, 1].set_title('IMU Info')
        self.ax[1, 1].set_xlabel('Time')

        if len(self.mode_time) % 2 == 1:
            self.mode_time.append(self.controltime[-1])
        for i in range(0, len(self.mode_time), 2):
            self.ax[0, 0].axvspan(
                self.mode_time[i], self.mode_time[i + 1], fc='0.1', alpha=0.1)
            self.ax[1, 0].axvspan(
                self.mode_time[i], self.mode_time[i + 1], fc='0.1', alpha=0.1)
            self.ax[1, 1].axvspan(
                self.mode_time[i], self.mode_time[i + 1], fc='0.1', alpha=0.1)
        plt.draw()

    def press(self, event):
        """
        Keyboard events during plotting
        """
        if event.key == 'q' or event.key == 'Q':
            plt.close('all')
        if event.key == 'a' or event.key == 'A':
            self.long()
        if event.key == 'z' or event.key == 'Z':
            self.lat()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description='Process and analyze control and planning data')
    parser.add_argument('--bag', help='Use Rosbag? (t/F)', default='F')
    args = parser.parse_args()

    rospy.init_node('control_info', anonymous=True)

    fig, axarr = plt.subplots(2, 2)
    plt.tight_layout()
    axarr[0, 0].get_shared_x_axes().join(axarr[0, 0], axarr[1, 0])
    axarr[1, 1].get_shared_x_axes().join(axarr[0, 0], axarr[1, 1])

    controlinfo = ControlInfo(axarr)

    if args.bag:
        file_path = args.bag
        bag = rosbag.Bag(file_path)
        for topic, msg, t in bag.read_messages(topics=[
                '/roadstar/control', '/roadstar/planning',
                '/roadstar/localization/pose', '/roadstar/canbus/chassis'
        ]):
            print t.to_sec(), topic
            if topic == "/roadstar/localization/pose":
                controlinfo.callback_localization(msg)
            elif topic == "/roadstar/planning":
                controlinfo.callback_planning(msg)
            elif topic == "/roadstar/control":
                controlinfo.callback_control(msg)
            elif topic == "/roadstar/canbus/chassis":
                controlinfo.callback_canbus(msg)
        print "Done reading the file"
        bag.close()

    else:
        planningsub = rospy.Subscriber('/roadstar/planning',
                                       planning_pb2.ADCTrajectory,
                                       controlinfo.callback_planning)
        localizationsub = rospy.Subscriber(
            '/roadstar/localization/pose', localization_pb2.LocalizationEstimate,
            controlinfo.callback_localization)
        controlsub = rospy.Subscriber('/roadstar/control',
                                      control_command_pb2.ControlCommand,
                                      controlinfo.callback_control)
        canbussub = rospy.Subscriber('/roadstar/canbus/chassis',
                                     chassis_pb2.Chassis,
                                     controlinfo.callback_canbus)
        controlinfo.update_subs(planningsub, localizationsub, controlsub,
                                canbussub, axarr)

        raw_input("Press Enter To Stop")

        planningsub.unregister()
        localizationsub.unregister()
        controlsub.unregister()
        canbussub.unregister()

        rospy.sleep(0.5)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    controlinfo.long()
    fig.canvas.mpl_connect('key_press_event', controlinfo.press)
    plt.show()


if __name__ == '__main__':
    main()
