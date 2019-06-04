import sys
import time
import modules.common.message.python as m
import modules.msgs.perception.proto.fusion_map_pb2 as fusion_map
from modules.common.message.tools.proto import message_bag_pb2 as message_bag
from modules.common.adapters.proto import adapter_config_pb2 as adapter_config

f = fusion_map.FusionMap()
f.obstacles.add().id = 10086

writer = m.BagWriter('/tmp/test.bag')
d = message_bag.BagDataChunk()
d.data_header.message_type = adapter_config.AdapterConfig.FUSION_MAP
d.data_header.receive_time_ns = 123
d.message_data = f.SerializeToString()
writer.FeedData(d)
writer.Close()

reader = m.BagReader(['/tmp/test.bag'])
d2 = message_bag.BagDataChunk()
while (reader.Next(d2)):
    if d2.data_header.message_type == adapter_config.AdapterConfig.FUSION_MAP:
        print d2
        f = fusion_map.FusionMap()
        f.ParseFromString(d2.message_data)
        print f
