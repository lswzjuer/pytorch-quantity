import sys
import time
import modules.common.message.python as m
import modules.msgs.perception.proto.fusion_map_pb2 as fusion_map
from modules.common.adapters.proto import adapter_config_pb2 as adapter_config


def cb(type, b, header_only):
    if type == adapter_config.AdapterConfig.FUSION_MAP:
        obj = fusion_map.FusionMap()
        obj.ParseFromString(b)
        print obj


m.Init(sys.argv, "planning", cb)

while True:
    time.sleep(1)
