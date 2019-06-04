#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "gflags/gflags.h"
#include "modules/common/log.h"
#include "modules/common/message/tools/utils.h"
#include "modules/common/message/tools/message_bag.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/message/tools/proto/message_bag.pb.h"

DEFINE_string(in_file, "", "imput filename");
DEFINE_string(out_file, "", "output filename");
DEFINE_string(types, "", "type to be extracted out");

using roadstar::common::message::BagReader;
using roadstar::common::message::BagWriter;
using roadstar::common::message::ParseTypeList;
using roadstar::common::message::BagDataChunk;
using roadstar::common::adapter::AdapterConfig;

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  roadstar::common::InitLogging(argv[0]);

  std::unordered_set<int> type_set, ignore_set;
  ParseTypeList(FLAGS_types, &type_set);
  for (int type = AdapterConfig::MessageType_MIN;
      type <= AdapterConfig::MessageType_MAX; type++) {
    if (!type_set.count(type)) {
        ignore_set.insert(type);
      }
  }

  BagReader reader(std::vector<std::string>{FLAGS_in_file}, ignore_set);
  BagWriter writer(FLAGS_out_file);
  BagDataChunk chunk;
  while (reader.Next(&chunk)) {
    if (type_set.count(chunk.data_header().message_type())) {
      writer.FeedData(chunk);
    }
  }
  
  return 0;
}
