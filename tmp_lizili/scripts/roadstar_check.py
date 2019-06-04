#!/usr/bin/python
import rospy
import thread
import sys
import os
import signal
import threading
import time
from enum import Enum
from std_msgs.msg import String
from modules.msgs.module_conf.proto import system_status_pb2
from google.protobuf import text_format
from modules.msgs.module_conf.proto import module_conf_pb2


class Status(Enum):
    DEPENDENCY = 0
    STARTING = 1
    OK = 2
    MOCK = 3
    ERROR = 4


module_conf_filename = 'config/modules/common/module_conf/conf/module_conf.pb.txt'
living_modules_filename = 'config/modules/common/module_conf/conf/living_modules_conf.pb.txt'
status_map = {
    Status.DEPENDENCY: "DEPENDENCY",
    Status.ERROR: "ERROR",
    Status.STARTING: "STARTING",
    Status.OK: "OK",
    Status.MOCK: "MOCK"
}

g_status_dict = {}
g_status_all_ok = {}
g_deps_dict = {}
g_command_dict = {}
g_status_dict_lock = threading.Lock()
g_exit_flag = 30
g_summary_flag = 0


def readFromProtoText(filename, object):
    f = open(filename, 'r')
    text = f.read()
    text_format.Merge(text, object)
    return object


def getDependencyDict(object):
    global g_deps_dict
    for living_module in object.living_modules:
        if living_module.name == "control":
            continue
        g_deps_dict[living_module.name] = list(living_module.dependency)


def executeCommand(command_str):
    global g_exit_flag
    g_exit_flag = 20
    ok = os.system("bash -c 'set -m && %s'" % command_str)
    if 1 == ok:
        print("execute Command success.")


def getCommandDict(object):
    for modules in object.modules:
        g_command_dict[modules.name] = modules.supported_commands


def callback(entity):
    global g_status_dict, g_summary_flag
    g_status_dict_lock.acquire()
    for key in entity.modules:
        if g_status_dict.has_key(key):
            if entity.modules[key].summary == system_status_pb2.OK:
                g_status_dict[key] = Status.OK
            elif entity.modules[key].summary == system_status_pb2.MOCK:
                g_status_dict[key] = Status.MOCK
    g_status_dict_lock.release()

    if entity.HasField('summary') and (
            entity.summary == system_status_pb2.OK
            or entity.summary == system_status_pb2.MOCK):
        g_summary_flag = 1


def listen():
    global g_exit_flag
    rospy.init_node('roadstar_check')
    rospy.Subscriber('/roadstar/monitor/system_status',
                     system_status_pb2.SystemStatus, callback)
    rate = rospy.Rate(1)
    loop_count = 0
    while True:
        if g_summary_flag == 1:
            print("All critical modules ok.")
            break
        elif g_exit_flag == 0:
            print("Timed out starting all modules")
            break
        g_exit_flag = g_exit_flag - 1
        rate.sleep()


def initStatusDict(object):
    global g_status_dict, g_status_all_ok
    for living_module in object.living_modules:
        if living_module.name == "control":
            continue
        g_status_dict[living_module.name] = Status.DEPENDENCY
        g_status_all_ok[living_module.name] = Status.OK


def stop_all():
    for module in g_deps_dict:
        if module != "dreamview" and module != "monitor":
            if module in g_command_dict and "stop" in g_command_dict[module]:
                executeCommand(g_command_dict[module]["stop"])
    executeCommand("scripts/control.sh stop")


def main():
    file_full_path = os.path.realpath(__file__)
    dir_full_path = os.path.dirname(file_full_path)
    slash = dir_full_path.rfind('/')
    project_absolute_path = dir_full_path[0:slash]

    # Set working directory
    os.chdir(project_absolute_path)

    # Load roadstar_config.py
    executeCommand("scripts/roadstar_config.py")
    # Load monitor.sh
    executeCommand("scripts/monitor.sh")

    living_modules_object = readFromProtoText(
        living_modules_filename, module_conf_pb2.LivingModuleSet())
    module_conf_object = readFromProtoText(module_conf_filename,
                                           module_conf_pb2.ModuleConfSet())
    initStatusDict(living_modules_object)
    getDependencyDict(living_modules_object)
    getCommandDict(module_conf_object)

    if "stop" in sys.argv:
        stop_all()
    else:
        thread.start_new_thread(threadFunc, ())
        listen()


def threadFunc():
    time.sleep(5)
    global g_status_dict
    while True:
        g_status_dict_lock.acquire()

        print(format("Module Running Status", "*^44"))
        print("--------------------------------------------\n")
        for key in g_status_dict:
            print("%s\t\t%s" % (format(key, "<20"),
                                status_map[g_status_dict[key]]))
        print("\n--------------------------------------------")

        for name in g_deps_dict:
            deps_all_ok = True
            for deps_module in g_deps_dict[name]:
                if g_status_dict[deps_module] != Status.OK and g_status_dict[deps_module] != Status.MOCK:
                    deps_all_ok = False
                    break

            if deps_all_ok == True and g_status_dict[name] == Status.DEPENDENCY:
                if 'start' in g_command_dict[name]:
                    print "Running %s" % g_command_dict[name]['start']
                    executeCommand(g_command_dict[name]['start'])
                    g_status_dict[name] = Status.STARTING
                    time.sleep(2)
                else:
                    g_status_dict[name] = Status.ERROR

        g_status_dict_lock.release()

        if g_status_all_ok == g_status_dict:
            break
        time.sleep(2)

    print("All modules launched.")


if __name__ == '__main__':
    try:
        signal.signal(signal.SIGINT, quit)
        signal.signal(signal.SIGTERM, quit)
        main()
    except KeyboardInterrupt as e:
        print(e.message)
