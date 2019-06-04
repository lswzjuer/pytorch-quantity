#!/usr/bin/env python


"""
Message Handle
"""
import curses
import rospy

import batch_include

Refreshrate = 16


class Message(object):
    """
    Message Class
    """

    def __init__(self, name, proto_name, topic, period, window, lock):
        self.name = name
        self.topic = topic
        self.lock = lock
        self.proto_name = 'batch_include.' + proto_name[:-2]
        self.proto = eval('batch_include.' + proto_name)
        self.period = float(period)

        self.msg_received = False
        self.msg_new = False
        self.msg_time = rospy.get_time()
        self.msg_delay = 0
        self.msg_interval = 0
        self.msg_max = 0
        self.sequence_num = 0
        self.msg_min = float("inf")
        self.field = Field(self.proto, window, self.proto.DESCRIPTOR)

        self.display_time = rospy.get_time()

    def callback(self, data):
        """
        callback function
        """
        nowtime = rospy.get_time()
        self.proto.CopyFrom(data)
        try:
            time = self.proto.header.timestamp_sec
            sequence_num = self.proto.header.sequence_num
        except:
            # seems no header or no timestamp in this proto data
            time = 0
            sequence_num = 0

        if self.msg_received == True:
            seq_diff = sequence_num - self.sequence_num
            if seq_diff is not 0:
                self.msg_interval = (time - self.msg_time) * 1000 / seq_diff
            else:
                self.msg_interval = (time - self.msg_time) * 1000
            if self.msg_interval > self.msg_max:
                self.msg_max = self.msg_interval
            if self.msg_interval < self.msg_min:
                self.msg_min = self.msg_interval
        self.msg_time = time
        self.sequence_num = sequence_num
        self.msg_delay = nowtime - time

        self.msg_received = True
        if self.field.show:
            nowtime = rospy.get_time()
            if (nowtime - self.display_time) > (1.0 / Refreshrate):
                with self.lock:
                    self.field.display_on_screen()
                    self.display_time = nowtime

    def key_up(self):
        """
        Keyboard Up Key
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.selection is not None:
            item.selection = max(item.selection - 1, 0)
        item.display_on_screen()

    def key_down(self):
        """
        Keyboard Down Key
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.selection is not None:
            item.selection = min(item.selection + 1, len(item.repeatedlist) - 1)
        item.display_on_screen()

    def key_right(self):
        """
        Keyboard Right Key
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.selection is not None:
            item.show = False
            item = item.repeatedlist[item.selection][3]
            item.show = True
        item.display_on_screen()

    def key_left(self):
        """
        Keyboard Left Key
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        item.show = False
        self.field.show = True
        self.field.display_on_screen()

    def index_incr(self):
        """
        Keyboard key to increment index in repeated item
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.index is not None:
            item.index = min(item.index + 1, len(item.item) - 1)
        item.display_on_screen()

    def index_decr(self):
        """
        Keyboard key to decrement index in repeated item
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.index is not None:
            item.index = max(item.index - 1, 0)
        item.display_on_screen()

    def index_begin(self):
        """
        Keyboard key to go to first element in repeated item
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.index is not None:
            item.index = 0
        item.display_on_screen()

    def index_end(self):
        """
        Keyboard key to go to last element in repeated item
        """
        item = self.field
        while item.show is False:
            item = item.repeatedlist[item.selection][3]
        if item.index is not None:
            item.index = len(item.item) - 1
        item.display_on_screen()


class Field(object):
    """
    Item in Message Class
    """

    def __init__(self, item, window, descriptor):
        self.repeatedlist = []
        self.item = item
        self.window = window
        self.show = False
        self.selection = None
        self.descriptor = descriptor
        self.index = None
        if descriptor.containing_type is not None and \
           descriptor.label == descriptor.LABEL_REPEATED:
            if len(item) != 0:
                self.index = 0

    def display_on_screen(self):
        """
        Display Wrapper
        """
        self.window.clear()
        self.windowy, self.windowx = self.window.getmaxyx()
        self.repeatedlist = []
        if self.descriptor.containing_type is not None and \
            self.descriptor.label == self.descriptor.LABEL_REPEATED:
            if self.index is not None:
                self.window.addstr(
                    0, 0, self.descriptor.name + ": " + str(self.index),
                    curses.A_BOLD)
                self.print_out(self.item[self.index], self.descriptor, 1, 2)
            else:
                self.window.addstr(0, 0, self.descriptor.name + ": Empty",
                                   curses.A_BOLD)
        else:
            self.window.addstr(0, 0, self.descriptor.name, curses.A_BOLD)
            self.print_out(self.item, self.descriptor, 1, 2)

        self.print_repeated()
        self.window.refresh()

    def print_out(self, entity, descriptor, row, col):
        """
        Handles display of each item in proto
        """
        if descriptor.containing_type is None  or \
           descriptor.type == descriptor.TYPE_MESSAGE:
            for descript, item in entity.ListFields():
                if row >= self.windowy:
                    if col >= (self.windowx / 3) * 2:
                        return row, col
                    row = 0
                    col = col + self.windowx / 3
                if descript.label == descript.LABEL_REPEATED:
                    printstring = descript.name + ": " + str(
                        len(item)) + "[Repeated Item]"
                    repeatedlist = [
                        col, row, printstring,
                        Field(item, self.window, descript)
                    ]
                    self.repeatedlist.append(repeatedlist)
                elif descript.type == descript.TYPE_MESSAGE:
                    self.window.addstr(row, col, descript.name + ": ")
                    row, col = self.print_out(item, descript, row + 1, col + 2)
                    row = row - 1
                    col = col - 2
                else:
                    self.print_out(item, descript, row, col)
                row = row + 1
            return row, col
        elif descriptor.type == descriptor.TYPE_ENUM:
            enum_type = descriptor.enum_type.values_by_number[entity].name
            self.window.addstr(row, col, descriptor.name + ": " + enum_type)
        elif descriptor.type == descriptor.TYPE_FLOAT or descriptor.type == descriptor.TYPE_DOUBLE:
            self.window.addstr(
                row, col, descriptor.name + ": " + "{0:.5f}".format(entity))
        else:
            self.window.addnstr(row, col, descriptor.name + ": " + str(entity),
                                self.windowx / 2)
        return row, col

    def print_repeated(self):
        """
        Special Handle for displaying repeated item
        """
        indx = 0
        if (len(self.repeatedlist) != 0 and self.selection is None) or \
            self.selection >= len(self.repeatedlist):
            self.selection = 0
        if len(self.repeatedlist) == 0:
            self.selection = None
        for item in self.repeatedlist:
            if indx == self.selection:
                self.window.addstr(item[1], item[0], item[2], curses.A_REVERSE)
            else:
                self.window.addstr(item[1], item[0], item[2], curses.A_BOLD)
            indx = indx + 1
