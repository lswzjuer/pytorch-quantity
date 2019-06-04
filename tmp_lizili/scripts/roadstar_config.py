#!/usr/bin/python3
import os
import time
import pwd
import shutil
import re
import socket
import sys
import subprocess
from fileinput import FileInput
from collections import defaultdict

CONFIG_TEMP_PATH = 'config.tmp.%f' % time.time()
CONFIG_PATH = 'config'
CONFIG_OLD_PATH = 'config_old'


def run_command(cmd):
    msgs = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True)
    msg = ""
    for tmp in msgs.stdout.readlines():
        msg += bytes.decode(tmp)
    return msg


vehicle = ["shiyan_truck", "hongqi", "trumpchi",
           "hongqibus", "howo", "simulation"]

# Vehicle-specific Maps
# Notice: Always add all truck keys when creating new maps!
conti_can_map = {
    "shiyan_truck1": "CHANNEL_ID_THREE",
    "shiyan_truck2": "CHANNEL_ID_THREE",
    "shiyan_truck3": "CHANNEL_ID_ONE",
    "shiyan_truck3.3": "CHANNEL_ID_ONE",
    "shiyan_truck4": "CHANNEL_ID_THREE",
    "shiyan_truck5": "CHANNEL_ID_THREE",
    "shiyan_truck_jenkins": "CHANNEL_ID_THREE",
    "hongqi0": "CHANNEL_ID_ONE",
    "hongqibus1": "CHANNEL_ID_ONE",
    "trumpchi1": "CHANNEL_ID_THREE",
    "howo1": "CHANNEL_ID_THREE",
    "simulation_powertrain": "CHANNEL_ID_THREE",
    "simulation_carla": "CHANNEL_ID_THREE",
}

vehicle_name_path = os.environ['HOME'] + "/.vehicle_name"
with open(vehicle_name_path, 'r') as f:
    vehicle_name = f.read().strip().replace("\n", "").replace("\r", "")

if vehicle_name.startswith("truck"):
    vehicle_name = "shiyan_" + vehicle_name

lidar_scheme_map = defaultdict(lambda: "truck_velo64_velo16")
lidar_scheme_map.update({
    "shiyan_truck1": "truck_hesai40_velo16",
    "shiyan_truck2": "truck_velo64_velo16",
    "shiyan_truck3": "truck_velo64_velo16",
    "shiyan_truck4": "truck_hesai40_velo16",
    "hongqi0": "car_velo64",
    "trumpchi1": "car_hesai40",
})

single_gpu = int(
    run_command(
        'nvidia-smi --query-gpu=gpu_name --format=csv | grep -v name | wc -l')
    .strip()) < 2

fake_vehicle = False
if re.split('\d+$|_[A-Za-z]+$', vehicle_name
            )[0] not in vehicle and vehicle_name != "shiyan_truck_jenkins":
    vehicle_name = "simulation_powertrain"
    fake_vehicle = True

drive_scene = os.environ.get('DRIVE_SCENE', default="undefined")
onboard = socket.gethostname().startswith("in_release_docker")

print("vehicle_name = %s, onboard = %d, drive_scene = %s, single_gpu = %d" %
      (vehicle_name, onboard, drive_scene, single_gpu))

# replace src if criteria matched.


def edit_line_with_regex(dst_path, line, criterias, from_regexs, tos):
    show_info = False
    if criterias is not None:
        for ind, criteria in enumerate(criterias):
            if criteria in line:
                line = re.sub(from_regexs[ind], tos[ind], line)
                show_info = True
    return line, show_info


def replace_and_append(src_path,
                       dst_path=None,
                       criterias=None,
                       from_regexs=None,
                       tos=None,
                       append=None):
    if not os.path.isfile(src_path):
        return
    if not dst_path:
        dst_path = os.path.join(CONFIG_TEMP_PATH, src_path)
    if criterias is not None:
        if not (len(criterias) == len(from_regexs) == len(tos)):
            raise Exception(
                "Lens of criterias, from_regexs and tos must be equal: %s" %
                (dst_path))
    if (os.path.islink(dst_path)):
        os.unlink(dst_path)
        src_file = open(src_path, 'r')
        conf_file = open(dst_path, 'w+')
        for line in src_file:
            origin = line
            line, show_info = edit_line_with_regex(dst_path, line, criterias,
                                                   from_regexs, tos)
            conf_file.write(line)
            if show_info:
                print(
                    "%s : %s -> %s" % (dst_path, origin.strip(), line.strip()))
        src_file.close()
        if append is not None:
            print("%s : %s" % (dst_path, append))
            conf_file.write(append)
        conf_file.close()
    else:
        cache_for_print = []
        with FileInput(files=[dst_path], inplace=True) as f:
            for line in f:
                origin = line
                line, show_info = edit_line_with_regex(
                    dst_path, line, criterias, from_regexs, tos)
                if show_info:
                    cache_for_print.append([origin, line])
                print(line.strip())
        for cache in cache_for_print:
            print("%s : %s -> %s" % (dst_path, cache[0].strip(),
                                     cache[1].strip()))
        if append is not None:
            with open(dst_path, 'a') as f:
                print("%s : %s" % (dst_path, append))
                f.write(append)


def vehicle_data_conf_setup():
    vehicle_data_path = os.path.join(CONFIG_TEMP_PATH, "vehicle_data")
    shutil.move(
        src=os.path.abspath(
            os.path.join(vehicle_data_path, vehicle_name + ".pb.txt")),
        dst=os.path.join(vehicle_data_path, "vehicle_in_use.pb.txt"))


def dreamview_conf_setup():
    src_path = os.path.join('modules', 'dreamview', 'conf', 'dreamview.conf')
    replace_and_append(
        src_path,
        criterias=["--calibration_config_path"],
        from_regexs=["shiyan_truck\d"],
        tos=[vehicle_name])


def perception_v2_conf_setup():
    config_path = 'modules/perception_v2/conf/perception.conf'
    lidar_config_dir = 'modules/perception_v2/conf/obstacle/lidar'
    onboard_dags = [
        'modules/perception_v2/conf/dag_city.config',
        'modules/perception_v2/conf/dag_highway.config',
        'modules/perception_v2/conf/dag_city_with_camera.config'
    ]
    regx_map = {
        "dag_config_path": "/dag_.*\.config",
        "onboard_segmentor": "CNNSegmentationV\d",
        "cnn_segmentation_v1_config": "cnnseg_v1_merge\.*_config.pb.txt",
        "cnn_segmentation_v2_config": "cnnseg_v2.*_config.pb.txt",
        "visible_range_x_backward_extra": "[0-9]\d*\.?\d*",
        "full_range": "(false|true)",
    }
    truck_config = {
        "trumpchi1": {
            "low_object_filter_config.pb.txt": ("full_range", "true"),
            "ray_segmentor_config.pb.txt": ("visible_range_x_backward_extra",
                                            "30"),
        }
    }

    lidar_city_model_catalog = {
        "truck_velo64_velo16": "cnnseg_v2_truck_velo64_velo16_config.pb.txt",
        "truck_hesai40_velo16": "cnnseg_v2_truck_hesai40_velo16_config.pb.txt",
        "car_velo64": "cnnseg_v2_truck_velo64_velo16_config.pb.txt",
        "car_hesai40": "cnnseg_v2_car_hesai40_config.pb.txt",
    }

    lidar_highway_model_catalog = {
        "truck_velo64_velo16": "cnnseg_v1_merge_config.pb.txt",
        "truck_hesai40_velo16": "cnnseg_v1_merge_config.pb.txt",
        "car_velo64": "cnnseg_v1_merge_config.pb.txt",
        "car_hesai40": "cnnseg_v1_merge_config.pb.txt",
    }

    drive_scene_config = {
        "city": {
            "dag_config_path": "/dag_city.config",
            "onboard_segmentor": "CNNSegmentationV2",
            "cnn_segmentation_v2_config":
            lidar_city_model_catalog[lidar_scheme_map[vehicle_name]],
        },
        "highway": {
            "dag_config_path": "/dag_highway.config",
            "onboard_segmentor": "CNNSegmentationV1",
            "cnn_segmentation_v1_config":
            lidar_highway_model_catalog[lidar_scheme_map[vehicle_name]],
        },
    }

    # calibration
    replace_and_append(
        config_path,
        criterias=["calibration_config_path"],
        from_regexs=["shiyan_truck\d"],
        tos=[vehicle_name])

    # truck specific options
    if vehicle_name in truck_config:
        config = truck_config[vehicle_name]
        for target_file, target in config.items():
            lidar_config_path = os.path.join(lidar_config_dir, target_file)
            replace_and_append(
                lidar_config_path,
                criterias=[target[0]],
                from_regexs=[regx_map[target[0]]],
                tos=[target[1]])

    # drive scene
    if drive_scene in drive_scene_config:
        config = drive_scene_config[drive_scene]
        for target, to in config.items():
            replace_and_append(
                config_path,
                criterias=[target],
                from_regexs=[regx_map[target]],
                tos=[to])

    # open camera object detection when scene=city and vehicle=truck1
    if onboard and drive_scene == "city" and vehicle_name == 'shiyan_truck1':
        replace_and_append(
            config_path,
            criterias=["dag_config_path"],
            from_regexs=[regx_map["dag_config_path"]],
            tos=["/dag_city_with_camera.config"])
        replace_and_append(
            config_path,
            criterias=["fuse_camera"],
            from_regexs=["false"],
            tos=["true"])

    # onboard dag
    for dag_path in onboard_dags:
        # onboard flags
        replace_and_append(
            dag_path,
            criterias=["onboard"],
            from_regexs=["onboard:\d"],
            tos=["onboard:" + str(int(onboard))])
        # gpu_id
        if single_gpu:
            replace_and_append(
                dag_path,
                criterias=["gpu_id"],
                from_regexs=["gpu_id:\d"],
                tos=["gpu_id:0"])


def vision_detector_fabu_conf_setup():
    src_path = 'modules/perception/camera/model/' \
        'vision_detector/vision_detector_fabu.conf'
    if onboard and single_gpu:
        replace_and_append(
            src_path, criterias=["gpu_id"], from_regexs=["\d"], tos=["0"])
    elif vehicle_name == "shiyan_truck_jenkins":
        replace_and_append(
            src_path, criterias=["gpu_id"], from_regexs=["\d"], tos=["1"])


def localization_conf_setup():
    if onboard:
        src_path = 'modules/localization/conf/' \
            'localization.conf'
        replace_and_append(
            src_path,
            criterias=["vehicle_name"],
            from_regexs=["shiyan_truck\d"],
            tos=[vehicle_name])


def highway_planning_conf_setup():
    if onboard:
        src_path = 'modules/planning/conf/' \
            'highway_planning_conf.txt'
        replace_and_append(
            src_path,
            criterias=["vehicle_name"],
            from_regexs=["shiyan_truck\d"],
            tos=[vehicle_name])
        src_path = 'modules/planning/conf/' \
            'urban_planning_conf.txt'
        replace_and_append(
            src_path,
            criterias=["vehicle_name"],
            from_regexs=["shiyan_truck\d"],
            tos=[vehicle_name])


def planning_conf_setup():
    config_path = 'modules/planning/conf/planning.conf'

    # vehicle info
    replace_and_append(
        config_path,
        criterias=["--vehicle_info_file"],
        from_regexs=["vehicle_in_use"],
        tos=[vehicle_name])

    # calibration file
    replace_and_append(
        config_path,
        criterias=["--imu2ego_calibration_file"],
        from_regexs=["vehicle_in_use"],
        tos=[vehicle_name])

    if onboard and vehicle_name == "shiyan_truck3.3":
        src_path = 'modules/planning/conf/planning.conf'
        replace_and_append(
            src_path, append='--hdmap_rpc_service_address=192.168.3.2:9999')


def control_param_setup():
    config_path = 'modules/control/conf/control.conf'

    # controller conf
    replace_and_append(
        config_path,
        criterias=["--control_conf_path"],
        from_regexs=["vehicle_in_use"],
        tos=[vehicle_name])

    # vehicle info
    replace_and_append(
        config_path,
        criterias=["--vehicle_info_file"],
        from_regexs=["vehicle_in_use"],
        tos=[vehicle_name])

    # calibration file
    replace_and_append(
        config_path,
        criterias=["--imu2ego_calibration_file"],
        from_regexs=["vehicle_in_use"],
        tos=[vehicle_name])


def simulation_node_setup():
    config_path = 'modules/simulation/simulation_world' \
        '/node/conf/simulation_world_node_main.conf'
    replace_and_append(
        config_path,
        criterias=["--calibration_config_path"],
        from_regexs=["shiyan_truck\d"],
        tos=[vehicle_name])


def dreamview_monitor_setup():
    if onboard:
        src_path = os.path.join('modules', 'dreamview', 'conf', 'hmi.conf')
        replace_and_append(
            src_path,
            criterias=["compressed_camera"],
            from_regexs=["compressed_camera"],
            tos=["camera"])


def monitor_setup():
    living_modules_conf_root_path = os.path.join(
        'resources', 'monitor', 'living_modules')
    monitor_conf_pb_txt = vehicle_name + "_living_modules.pb.txt"
    src_path = os.path.join(living_modules_conf_root_path,
                            monitor_conf_pb_txt)
    dst_path = os.path.join(CONFIG_TEMP_PATH, 'modules', 'common',
                            'module_conf', 'conf',
                            'living_modules_conf.pb.txt')

    shutil.copy2(src_path, dst_path)
    if not onboard:
        replace_and_append(
            src_path,
            dst_path=dst_path,
            criterias=["CAMERA"],
            from_regexs=["CAMERA"],
            tos=["COMPRESSED_CAMERA"])
        replace_and_append(
            src_path,
            dst_path=dst_path,
            criterias=["endpoint"],
            from_regexs=["\d+\.\d+\.\d+\.\d+"],
            tos=["127.0.0.1"])

        simulation_living_modules_path = os.path.join(
            living_modules_conf_root_path, 'simulation_living_modules.pb.txt')
        modules_str = open(simulation_living_modules_path).read()
        replace_and_append(src_path, dst_path=dst_path, append=modules_str)


def vehicle_conf_setup():
    if fake_vehicle:
        return
    if onboard:
        src_path = "modules/vehicle/conf/vehicle_conf.pb.txt"
        if re.match('shiyan_truck[1-2]', vehicle_name) is not None:
            replace_and_append(
                src_path,
                criterias=["vehicle_brand"],
                from_regexs=[":\s\w*$"],
                tos=[": DONGFENG"])
            replace_and_append(
                src_path,
                criterias=["bitrate"],
                from_regexs=[":\s\d{3}$"],
                tos=[": 250"])
            replace_and_append(
                src_path,
                criterias=["use_extended_frame"],
                from_regexs=[":\s\w*$"],
                tos=[": true"])
        elif re.match('shiyan_truck[3]', vehicle_name) is not None:
            replace_and_append(
                src_path,
                criterias=["vehicle_brand"],
                from_regexs=[":\s\w*$"],
                tos=[": DONGFENG_V1"])
            replace_and_append(
                src_path,
                criterias=["bitrate"],
                from_regexs=[":\s\d{3}$"],
                tos=[": 250"])
            replace_and_append(
                src_path,
                criterias=["use_extended_frame"],
                from_regexs=[":\s\w*$"],
                tos=[": true"])
        elif re.match('shiyan_truck[4]', vehicle_name) is not None:
            replace_and_append(
                src_path,
                criterias=["vehicle_brand"],
                from_regexs=[":\s\w*$"],
                tos=[": DONGFENG_V2"])
            replace_and_append(
                src_path,
                criterias=["bitrate"],
                from_regexs=[":\s\d{3}$"],
                tos=[": 250"])
            replace_and_append(
                src_path,
                criterias=["use_extended_frame"],
                from_regexs=[":\s\w*$"],
                tos=[": true"])
        elif re.match('hongqi[0-2]', vehicle_name) is not None:
            replace_and_append(
                src_path,
                criterias=["vehicle_brand"],
                from_regexs=[":\s\w*$"],
                tos=[": HONGQI"])
            replace_and_append(
                src_path,
                criterias=["bitrate"],
                from_regexs=[":\s\d{3}$"],
                tos=[": 500"])
            replace_and_append(
                src_path,
                criterias=["use_extended_frame"],
                from_regexs=[":\s\w*$"],
                tos=[": false"])
        elif re.match('trumpchi[0-1]', vehicle_name) is not None:
            replace_and_append(
                src_path,
                criterias=["vehicle_brand"],
                from_regexs=[":\s\w*$"],
                tos=[": TRUMPCHI"])
            replace_and_append(
                src_path,
                criterias=["bitrate"],
                from_regexs=[":\s\d{3}$"],
                tos=[": 500"])
            replace_and_append(
                src_path,
                criterias=["use_extended_frame"],
                from_regexs=[":\s\w*$"],
                tos=[": false"])
        elif re.match('hongqibus[0-1]', vehicle_name) is not None:
            replace_and_append(
                src_path,
                criterias=["vehicle_brand"],
                from_regexs=[":\s\w*$"],
                tos=[": HONGQI_V2"])
            replace_and_append(
                src_path,
                criterias=["bitrate"],
                from_regexs=[":\s\d{3}$"],
                tos=[": 500"])
            replace_and_append(
                src_path,
                criterias=["use_extended_frame"],
                from_regexs=[":\s\w*$"],
                tos=[": false"])


def conti_radar_driver_setup():
    if onboard:
        src_path = "modules/drivers/conti_radar/conf/" \
                   "conti_radar_head_mid_conf.pb.txt"
        replace_and_append(
            src_path,
            criterias=["channel_id"],
            from_regexs=["CHANNEL_ID_ONE"],
            tos=[conti_can_map[vehicle_name]])

        src_path = "modules/drivers_v2/radar/conti_radar/conf/" \
                   "conti_radar_head_mid_config.pb.txt"
        replace_and_append(
            src_path,
            criterias=["channel_id"],
            from_regexs=["CHANNEL_ID_ONE"],
            tos=[conti_can_map[vehicle_name]])


def radar_driver_setup():
    if fake_vehicle:
        return
    src_path = "modules/drivers_v2/radar/conf/radar_driver.conf"
    if re.match('howo[1]', vehicle_name) is not None:
        replace_and_append(
            src_path,
            criterias=["--dag_config_path"],
            from_regexs=["vehicle_in_use"],
            tos=[vehicle_name])
    else:
        replace_and_append(
            src_path,
            criterias=["--dag_config_path"],
            from_regexs=["vehicle_in_use/"],
            tos=[""])


def lidar_driver_setup():
    if fake_vehicle:
        return
    src_path = "modules/drivers_v2/lidar/conf/lidar_driver.conf"
    dst = "resources/drivers/lidar/{}".format(vehicle_name)
    if not onboard:
        replace_and_append(
            src_path,
            criterias=["--onboard"],
            from_regexs=["true"],
            tos=["false"])

    replace_and_append(
        src_path,
        criterias=["--dag_config_path"],
        from_regexs=["config/modules/drivers_v2/lidar/conf"],
        tos=[dst])

    replace_and_append(
        src_path,
        criterias=["--lidar_conf_root"],
        from_regexs=["config/modules/drivers_v2/lidar/conf"],
        tos=[dst])


def camera_driver_setup():
    if onboard and vehicle_name in ['shiyan_truck1',
                                    'shiyan_truck2',
                                    'shiyan_truck3',
                                    'shiyan_truck4']:
        replace_and_append('modules/drivers_v2/camera/conf/camera_driver.conf',
                           append="--basler_start_with=head_left,front_right")
    elif onboard and vehicle_name == 'shiyan_truck5':
        replace_and_append('modules/drivers_v2/camera/conf/camera_driver.conf',
                           append="--basler_start_with=head_left,"
                           "head_right, front_left, front_right")
    else:
        replace_and_append('modules/drivers_v2/camera/conf/camera_driver.conf',
                           append="--basler_start_with=head_left,head_right")


def make_symlink(src, dst):
    if src.endswith(
            (".conf", ".txt", ".config", '.yaml', '.csv')):
        os.symlink(os.path.abspath(src), dst)


def copy_config(src, dst):
    shutil.copytree(src, dst, copy_function=make_symlink)


def conf_setup():
    if os.path.isdir(CONFIG_TEMP_PATH):
        shutil.rmtree(CONFIG_TEMP_PATH)
    os.makedirs(CONFIG_TEMP_PATH)
    copy_config('modules', os.path.join(CONFIG_TEMP_PATH, 'modules'))
    copy_config(
        os.path.join('resources', 'vehicle_data'),
        os.path.join(CONFIG_TEMP_PATH, 'vehicle_data'))

    vehicle_data_conf_setup()
    perception_v2_conf_setup()
    localization_conf_setup()
    highway_planning_conf_setup()
    planning_conf_setup()
    vision_detector_fabu_conf_setup()
    control_param_setup()
    dreamview_conf_setup()
    monitor_setup()
    vehicle_conf_setup()
    conti_radar_driver_setup()
    radar_driver_setup()
    lidar_driver_setup()
    camera_driver_setup()
    simulation_node_setup()

    if os.path.isdir(CONFIG_OLD_PATH):
        shutil.rmtree(CONFIG_OLD_PATH)
    if os.path.isdir(CONFIG_PATH):
        shutil.move(CONFIG_PATH, CONFIG_OLD_PATH)
    shutil.move(CONFIG_TEMP_PATH, CONFIG_PATH)
    if os.path.isdir(CONFIG_OLD_PATH):
        shutil.rmtree(CONFIG_OLD_PATH)


def main():
    print('Creating config for %s' % vehicle_name)
    os.chdir(os.path.join(os.path.dirname(sys.argv[0]), '..'))
    conf_setup()


if __name__ == "__main__":
    main()
