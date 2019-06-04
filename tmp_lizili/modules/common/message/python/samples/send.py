import sys
import time
import modules.common.message.python as m
import modules.msgs.perception.proto.fusion_map_pb2 as fusion_map
from modules.common.adapters.proto import adapter_config_pb2 as adapter_config


def cb(type, b, header_only):
    pass


m.Init(sys.argv, "perception_v2", cb)

while True:
    obj = fusion_map.FusionMap()
    obj.obstacles.add().id = 10086
    m.Send(adapter_config.AdapterConfig.FUSION_MAP, obj.SerializeToString())
    time.sleep(1)
