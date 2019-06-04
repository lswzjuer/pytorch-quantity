import os
import rospy
import random
import pprint
import time
from message import Message
from batch_include import topic_infos
from std_msgs.msg import String

RefresTime = 0.2  # unit:second

TopicWatched_truck0 = [
    'CAMERA_HL',
    'CAMERA_HR',
    'CAMERA_FL',
    'CAMERA_FR',
    'CAMERA_TL',
    'CAMERA_TR',
    'LIDAR_MAIN',
    'LIDAR_HM',
    'LIDAR_TL',
    'LIDAR_TR',
    'RADAR_HM',
    'RADAR_TM',
    'CANBUS',
    'LOC',  # localization
    'PERCEP',
]

vehicle_config = {
    'truck0': TopicWatched_truck0,
    'truck1': TopicWatched_truck0,
    'truck2': TopicWatched_truck0,
    'truck3': TopicWatched_truck0,
    'truck4': TopicWatched_truck0,
}

vehicle_name_path = os.environ['HOME'] + "/.vehicle_name"
with open(vehicle_name_path, 'r') as f:
    vehicle_name = f.read().strip().replace("\n", "").replace("\r", "")
    print "VEHICLE NAME: ", vehicle_name
if vehicle_config.has_key(vehicle_name):
    #print vehicle_name
    pass
else:
    print "******"
    print "******"
    print "** ERROR: ~/.vehicle_name is not set or not in vehicles list"
    print "******"
    print "******"
    print "** SET vehicle_name: truck0"
    print "******"
    print "******"
    vehicle_name = 'truck0'

messages = []
rospy.init_node('adu_diagnostics_' + str(random.random()), anonymous=True)
for i, topic_info in enumerate(topic_infos):
    if not (topic_info['module_name'] in vehicle_config[vehicle_name]):
        continue
    module_name = topic_info['module_name']
    topic_name = topic_info['topic_name']
    proto = topic_info['proto']
    hz_mean = float(topic_info['hz_mean'])
    hz_dev = float(topic_info['hz_dev'])
    cur_message = Message(module_name, proto, topic_name, hz_mean, hz_dev)
    messages.append(cur_message)

sublist = []
update_flags = []
for msg in messages:
    sublist.append(
        rospy.Subscriber(msg.topic, msg.proto, msg.callback, queue_size=100))
    update_flags.append(False)

last_msg_infos = []
while not rospy.is_shutdown():

    msg_infos = []
    for i, msg in enumerate(messages):
        msg_infos.append(msg.get_msg_info())
        if not update_flags[i]:
            messages[i].clear()

    #pprint.pprint(msg_infos)
    output_str_line0 = '--------|'
    output_str_line1 = '  TOPIC:|'
    output_str_line2 = '     HZ:|'
    #output_str_line3 = ' PERIOD:|'
    #output_str_line4 = 'RECEIVED: |'
    output_str_line5 = 'UPDATED:|'
    output_str_line3 = ' STATUS:|'
    output_str_line6 = '--------|'
    for i, msg_info in enumerate(msg_infos):
        output_str_line0 = output_str_line0 + '-----------|'
        output_str_line1 = output_str_line1 + '%10s |' % (
            msg_info['module_name'])
        #output_str_line3 = output_str_line3 + '%10d |' % (msg_info['msg_interval'])
        #output_str_line4 = output_str_line4 + ' %11d |' % (str(msg_info['msg_received']))
        if msg_info['counter'] >= 1:
            if last_msg_infos[i]['counter'] == msg_info['counter']:
                output_str_line5 = output_str_line5 + '\033[1;31;40m%10s\033[0m |' % (
                    'False')
                output_str_line2 = output_str_line2 + '\033[1;37;40m%10.2f\033[0m |' % (
                    0.0)
                update_flags[i] = False
            else:
                output_str_line5 = output_str_line5 + '\033[1;37;40m%10s\033[0m |' % (
                    'True')
                output_str_line2 = output_str_line2 + '\033[1;37;40m%10.2f\033[0m |' % (
                    msg_info['final_hz'])
                update_flags[i] = True
        else:
            output_str_line5 = output_str_line5 + '\033[1;31;40m%10s\033[0m |' % (
                'False')
            output_str_line2 = output_str_line2 + '\033[1;37;40m%10.2f\033[0m |' % (
                0.0)
            update_flags[i] = False

        status_string = 'Bad'
        if update_flags[i]:
            if msg_info['final_hz'] < msg_info['hz_mean'] + msg_info['hz_dev'] and msg_info['final_hz'] > msg_info['hz_mean'] - msg_info['hz_dev']:
                status_string = 'Good'
                output_str_line3 = output_str_line3 + '\033[1;37;40m%10s\033[0m |' % (
                    status_string)
            else:
                status_string = 'Bad'
                output_str_line3 = output_str_line3 + '\033[1;31;42m%10s\033[0m |' % (
                    status_string)
        else:
            status_string = 'Bad'
            output_str_line3 = output_str_line3 + '\033[1;31;42m%10s\033[0m |' % (
                status_string)

        output_str_line6 = output_str_line6 + '-----------|'

    #output_str_line = output_str_line0 + '\n' + output_str_line1 + '\n' + output_str_line2 + '\n' + output_str_line3 + '\n' + output_str_line4 + '\n' + output_str_line5 + '\n' + output_str_line6 + '\n'
    output_str_line = output_str_line0 + '\n' + output_str_line1 + '\n' + output_str_line2 + \
        '\n' + output_str_line5 + '\n' + output_str_line3 + '\n' + output_str_line6 + '\n'
    #output_str_line = output_str_line0 + '\n' + output_str_line1 + '\n' + output_str_line2 + '\n' + output_str_line5 + '\n' + output_str_line6 + '\n'
    print output_str_line

    last_msg_infos = msg_infos
    rospy.sleep(1.0)
#rospy.spin()
