#!/usr/bin/env python


"""
Message Handle
"""
import curses
import rospy
import copy

from collections import deque

import batch_include

DequeMaxLength = 500


class Message(object):
    """
    Message Class
    """

    def __init__(self, name, proto_name, topic, hz_mean, hz_dev):
        self.name = name
        self.topic = topic
        #self.lock = lock
        self.proto_name = proto_name
        #self.proto = eval(proto_name)
        self.proto = proto_name
        self.hz_mean = float(hz_mean)
        self.hz_dev = float(hz_dev)

        self.msg_received = False
        self.msg_new = False
        self.msg_time = rospy.get_time()
        self.msg_delay = 0
        self.msg_interval = 0
        self.msg_max = 0
        self.sequence_num = 0
        self.msg_min = float("inf")
        #self.field = Field(self.proto, window, self.proto.DESCRIPTOR)

        self.display_time = rospy.get_time()
        self.counter = int(-1)
        self.hz = 0.0
        self.timestamp_deque = deque(maxlen=DequeMaxLength)
        self.final_hz = 0.0

    def callback(self, data):
        """
        callback function
        """
        nowtime = rospy.get_time()
        #self.proto.CopyFrom(data)
        self.proto = data
        self.counter = self.counter + 1
        try:
            time = self.proto.header.timestamp_sec
            #sequence_num = self.proto.header.sequence_num
        except:
            # seems no header or no timestamp in this proto data
            time = nowtime
            #sequence_num = self.counter
        #time = nowtime
        sequence_num = self.counter
        self.timestamp_deque.append(time)

        if self.msg_received == True:
            seq_diff = sequence_num - self.sequence_num
            '''
            if seq_diff is not 0:
                self.msg_interval = (time - self.msg_time) * 1000 / seq_diff
            else:
                self.msg_interval = (time - self.msg_time) * 1000
            '''
            self.msg_interval = (time - self.msg_time) * 1000.0
            if self.msg_interval <= 1e-3:
                self.hz = 1e3
            else:
                self.hz = 1000 / self.msg_interval
            if self.msg_interval > self.msg_max:
                self.msg_max = self.msg_interval
            if self.msg_interval < self.msg_min:
                self.msg_min = self.msg_interval
            #print self.timestamp_deque
            self.final_hz = self.calculate_final_hz(self.timestamp_deque)
            #print self.final_hz

        self.msg_time = time
        self.sequence_num = sequence_num
        self.msg_delay = nowtime - time
        self.msg_received = True

    def clear(self):
        self.msg_received = False
        self.msg_new = False
        self.msg_time = rospy.get_time()
        self.msg_delay = 0
        self.msg_interval = 0
        self.msg_max = 0
        self.sequence_num = 0
        self.msg_min = float("inf")
        #self.field = Field(self.proto, window, self.proto.DESCRIPTOR)

        self.display_time = rospy.get_time()
        self.counter = int(-1)
        self.hz = 0.0
        self.timestamp_deque = deque(maxlen=DequeMaxLength)
        self.final_hz = 0.0

    def get_msg_info(self):
        msg_info = {}
        msg_info['msg_received'] = self.msg_received
        msg_info['msg_interval'] = self.msg_interval
        msg_info['msg_max'] = self.msg_max
        msg_info['msg_min'] = self.msg_min
        msg_info['hz'] = self.hz
        msg_info['hz_mean'] = self.hz_mean
        msg_info['hz_dev'] = self.hz_dev
        msg_info['final_hz'] = self.final_hz
        msg_info['counter'] = self.counter
        msg_info['module_name'] = self.name
        return copy.deepcopy(msg_info)

    def calculate_final_hz(self, timestamp_deque):  # calculate mean of hz
        deque_size = len(timestamp_deque)
        final_hz = 0.0
        if deque_size <= 1:
            return final_hz
        else:
            delta_time = []
            for i in range(deque_size):
                if i > 0:
                    delta_time.append(
                        timestamp_deque[i] - timestamp_deque[i - 1])
            if sum(delta_time) < 1e-3:
                final_hz = 10000
            else:
                final_hz = 1.0 / (sum(delta_time) / float(len(delta_time)))
        return final_hz
