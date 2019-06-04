#include <endian.h>
#include <time.h>
#include <csignal>
#include <vector>

#include "gflags/gflags.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/log.h"
#include "modules/common/message/message_service.h"
#include "modules/common/message/tools/bag_tool.h"
#include "modules/common/message/tools/proto/message_bag.pb.h"
#include "modules/common/message/tools/utils.h"

DEFINE_string(save_bag_path, "", "path to save bag to");
DEFINE_int32(bag_interval_seconds, 60, "number of seconds in splitted bag");
DEFINE_int32(buffer_size, 10000, "buffer size");

using roadstar::common::adapter::AdapterConfig;
using roadstar::common::message::BagDataChunk;
using roadstar::common::message::BagRecorder;
using roadstar::common::message::GetTimestampNs;
using roadstar::common::message::MessageService;
using roadstar::common::message::SleepUntil;

namespace {

std::unique_ptr<BagRecorder> recorder;

void SignalHandler(int num) {
  if (recorder) {
    recorder->Close();
    recorder.reset();
  }
}

void Callback(const AdapterConfig::MessageType type,
              const std::vector<unsigned char> &buffer, bool header_only) {
  if (header_only) {
    AFATAL << "record bag receiving header_only message : "
           << AdapterConfig::MessageType_descriptor()
                  ->FindValueByNumber(type)
                  ->name();
  }
  ADEBUG << "msg received :"
         << AdapterConfig::MessageType_descriptor()
                ->FindValueByNumber(type)
                ->name();

  BagDataChunk chunk;
  chunk.mutable_data_header()->set_message_type(type);
  chunk.mutable_data_header()->set_receive_time_ns(GetTimestampNs());
  chunk.set_message_data(&buffer[0], buffer.size());
  recorder->FeedData(chunk);
}

void OpenBag() {
  static int bag_order = 0;
  if (!recorder) {
    recorder.reset(new BagRecorder(FLAGS_buffer_size));
  }

  std::time_t t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  char buffer[200];
  strftime(buffer, sizeof(buffer), "%Y_%m_%d_%H_%M_%S", &tm);
  std::string filename = FLAGS_save_bag_path + "_" + buffer + "_" +
                         std::to_string(bag_order) + ".msg";

  recorder->Open(filename);
  bag_order++;
}
}  // namespace

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  roadstar::common::InitLogging(argv[0]);

  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  MessageService::Init("record_bag", Callback);
  uint64_t next_bag_time = GetTimestampNs();
  while (true) {
    OpenBag();
    next_bag_time += FLAGS_bag_interval_seconds * 1000000000ll;
    SleepUntil(next_bag_time);
  }
}
