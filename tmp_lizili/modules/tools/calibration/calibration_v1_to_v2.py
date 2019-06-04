#!/usr/bin/env python
"""
Change calibration confs form v1 to v2, all sensor2imu
"""
import glob
import os
from argparse import ArgumentParser
import numpy as np


def euler_to_affine(trans_dict):
    """
    Euler angle and translation to affine matrix
    """
    # R*P + t, z: heading, y: pitch, x: roll
    # R = Rz * Ry * Rx
    heading = trans_dict['heading']
    pitch = trans_dict['pitch']
    roll = trans_dict['roll']
    trans_xyz = [trans_dict['x'], trans_dict['y'], trans_dict['z']]
    affine_matrix = np.zeros([4, 4])
    rot_z = np.array([[np.cos(heading), -np.sin(heading), 0],
                      [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    rot_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
    affine_matrix[0:3, 0:3] = rot_matrix
    affine_matrix[:3, 3], affine_matrix[3, 3] = trans_xyz, 1
    return affine_matrix


def affine_to_euler(affine_matrix):
    """
    Affine matrix to euler angle and translation
    """
    # R*P + t, z: heading, y: pitch, x: roll
    # R = Rz * Ry * Rx
    trans_dict = {
        'x': 0.,
        'y': 0.,
        'z': 0.,
        'heading': 0.,
        'pitch': 0.,
        'roll': 0.
    }
    trans_dict['x'] = affine_matrix[0, 3]
    trans_dict['y'] = affine_matrix[1, 3]
    trans_dict['z'] = affine_matrix[2, 3]
    trans_dict['roll'] = np.arctan2(affine_matrix[2, 1], affine_matrix[2, 2])
    trans_dict['pitch'] = np.arctan2(-affine_matrix[2, 0],
                                     np.linalg.norm(affine_matrix[2, 1:3]))
    trans_dict['heading'] = np.arctan2(affine_matrix[1, 0],
                                       affine_matrix[0, 0])
    return trans_dict


def read_conf(conf_file, param_name_update):

    param_dict = {
        'x': 0.,
        'y': 0.,
        'z': 0.,
        'heading': 0.,
        'pitch': 0.,
        'roll': 0.
    }
    with open(conf_file, 'r') as in_file:
        lines = in_file.readlines()
    lines = [line.rstrip() for line in lines]
    in_dict = {}
    for line in lines:
        items = line.split(':')
        if len(items) == 2:
            if items[0] in param_name_update.keys():
                v2_param_name = param_name_update[items[0]]
                param_dict[v2_param_name] = float(items[1])
    return param_dict


def write_conf(trans_dict, conf_file):

    with open(conf_file, 'w') as out_file:
        for key, value in sorted(trans_dict.items()):
            value = float("{:.6f}".format(trans_dict[key]))
            out_file.write(key + ": " + str(value) + "\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        dest="input_path",
        required=True,
        metavar="input_path",
        help="input path containing v1 (sensor2ego) confs")
    parser.add_argument(
        "-o",
        dest="output_path",
        required=True,
        metavar="output_path",
        help="output path for v2 (sensor2imu) confs")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    param_name_update = {
        "transform_x": "x",
        "esr_x": "x",
        "x": "x",
        "transform_y": "y",
        "esr_y": "y",
        "y": "y",
        "transform_z": "z",
        "esr_z": "z",
        "z": "z",
        "transform_theta": "heading",
        "esr_phi": "heading",
        "theta": "heading",
        "pitch": "pitch",
        "roll": "roll"
    }

    # v1 confs
    imu_conf = "imu_coords_calibration.conf"  # to ego
    vlp16_confs_list = [
        "vlp_calibration0.conf",  # to vlp64/40
        "vlp_calibration1.conf",
        "vlp_calibration2.conf"
    ]
    vlp_conf = "velodyne_calibration.conf"  # 64/40 lidar, to ego
    conti_front_conf = "conti_calibration_front.conf"  # need check
    conti_tail_conf = "conti_calibration.conf"
    radar_conf = "radar_install.conf"

    conf_name_update = {
        imu_conf: "ego_front.conf",
        conti_tail_conf: "radar_tail_mid.conf",
        radar_conf: "radar_head_mid.conf",
        conti_front_conf: "radar_head_mid.conf",
        vlp_conf: "main_lidar_calibration.conf",
        vlp16_confs_list[0]: "vlp_calibration0.conf",
        vlp16_confs_list[1]: "vlp_calibration1.conf",
        vlp16_confs_list[2]: "vlp_calibration2.conf"
    }

    conf_list = glob.glob(os.path.join(args.input_path, "*.conf"))
    conf_dict = {os.path.basename(item): item for item in conf_list}

    affine_dict = {}
    if imu_conf in conf_dict.keys():
        param_dict = read_conf(conf_dict[imu_conf], param_name_update)
        # !!! -heading for imu
        param_dict['heading'] = -param_dict['heading']
        affine_dict[imu_conf] = euler_to_affine(param_dict)

    if vlp_conf in conf_dict.keys():
        param_vlp_dict = read_conf(conf_dict[vlp_conf], param_name_update)
        affine_dict[vlp_conf] = euler_to_affine(param_vlp_dict)
        for vlp16_conf in vlp16_confs_list:
            if vlp16_conf in conf_dict.keys():
                param_dict = read_conf(conf_dict[vlp16_conf],
                                       param_name_update)
                # !!! z is to ego
                param_dict["z"] -= param_vlp_dict["z"]
                affine_dict[vlp16_conf] = np.dot(affine_dict[vlp_conf],
                                                 euler_to_affine(param_dict))
    # !!! only one for radar_head_mid
    if conti_front_conf in conf_dict.keys():
        param_dict = read_conf(conf_dict[conti_front_conf], param_name_update)
        affine_dict[conti_front_conf] = euler_to_affine(param_dict)
    elif radar_conf in conf_dict.keys():
        param_dict = read_conf(conf_dict[radar_conf], param_name_update)
        affine_dict[radar_conf] = euler_to_affine(param_dict)

    if conti_tail_conf in conf_dict.keys():
        param_dict = read_conf(conf_dict[conti_tail_conf], param_name_update)
        affine_dict[conti_tail_conf] = euler_to_affine(param_dict)

    if imu_conf in affine_dict.keys():
        trans_ego2imu = np.linalg.inv(affine_dict[imu_conf])
        for conf_name, trans_sensor2ego in affine_dict.items():
            if conf_name != imu_conf:
                trans_sensor2imu = np.dot(trans_ego2imu, trans_sensor2ego)
            else:
                trans_sensor2imu = trans_ego2imu
            v2_conf_name = os.path.join(args.output_path,
                                        conf_name_update[conf_name])
            write_conf(affine_to_euler(trans_sensor2imu), v2_conf_name)
    else:
        print("imu_coords_calibration.conf not found:")
