#!/usr/bin/env python



from modules.canbus.proto import chassis_detail_pb2
from modules.canbus.proto import chassis_pb2
from modules.common.monitor.proto import monitor_pb2
from modules.common.configs.proto import config_extrinsics_pb2
from modules.common.configs.proto import vehicle_config_pb2
from modules.common.proto import geometry_pb2
from modules.common.proto import header_pb2
from modules.control.proto import control_command_pb2
from modules.control.proto import pad_msg_pb2
from modules.decision.proto import decision_pb2
from modules.localization.proto import localization_pb2
from modules.localization.proto import gps_pb2
from modules.localization.proto import imu_pb2
from modules.perception.proto import perception_obstacle_pb2
from modules.perception.proto import traffic_light_detection_pb2
from modules.planning.proto import planning_internal_pb2
from modules.planning.proto import planning_pb2
from modules.prediction.proto import prediction_obstacle_pb2
