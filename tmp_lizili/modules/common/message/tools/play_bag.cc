#include <endian.h>
#include <time.h>
#include <csignal>

#include "gflags/gflags.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/log.h"
#include "modules/common/message/message_service.h"
#include "modules/common/message/tools/bag_tool.h"
#include "modules/common/message/tools/utils.h"

DEFINE_bool(i, false, "only show bag info");
DEFINE_bool(l, false, "loop playback");
DEFINE_double(s, 0, "time start to play");
DEFINE_double(u, -1, "play duration time");
DEFINE_double(r, 1, "play speed rate");
DEFINE_string(ignore, "",
              "ignore specific types of message, eg: -ignore TYPE1,TYPE2,...");

using roadstar::common::adapter::AdapterConfig;
using roadstar::common::message::BagDataChunk;
using roadstar::common::message::BagPlayer;
using roadstar::common::message::GetTimestampNs;
using roadstar::common::message::KeyboardListener;
using roadstar::common::message::MessageService;
using roadstar::common::message::NanoSecondToSecond;
namespace {
void Callback(const AdapterConfig::MessageType type,
              const std::vector<unsigned char> &buffer, bool header_only) {}

void ShowProgressBar(uint64_t chunk_time_ns, uint64_t start_time_ns,
                     uint64_t end_time_ns) {
  static uint64_t last_show = 0;
  if (chunk_time_ns == end_time_ns) {
    last_show = 0;
  }
  uint64_t now = GetTimestampNs();
  if (now - last_show < 50000000ll) {  // 20 fps
    return;
  }
  last_show = now;

  double palyed = NanoSecondToSecond(chunk_time_ns - start_time_ns);
  double duration = NanoSecondToSecond(end_time_ns - start_time_ns);
  std::cout << " [PLAYBAG] ";
  std::cout << " Duration: " << std::fixed << palyed << " / " << std::fixed
            << duration << "    \r" << std::flush;
}

}  // namespace

int main(int argc, char *argv[]) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc <= 1) {
    return 0;
  }
  std::vector<std::string> filenames;
  std::transform(argv + 1, argv + argc, std::back_inserter(filenames),
                 [](char *s) {
                   std::ifstream fin(s);
                   AFATAL_IF(!fin) << "Can not open file " << s;
                   return std::string(s);
                 });

  auto player = BagPlayer(filenames, FLAGS_ignore);
  player.ShowBagInfo();
  if (FLAGS_i) {
    exit(0);
  }

  auto keyboard_listener = KeyboardListener();
  keyboard_listener.RegisterKey(' ', [&player]() { player.TogglePaused(); });
  keyboard_listener.RegisterKey('s', [&player]() { player.StepAhead(); });
  keyboard_listener.StartListen();

  MessageService::Init("play_bag", Callback);
  std::cout << " \nHit space to toggle paused, or 's' to step." << std::endl;
  player.Launch(FLAGS_s, FLAGS_u, FLAGS_r, FLAGS_l,
                [](const BagDataChunk &chunk, uint64_t start_time_ns,
                   uint64_t end_time_ns) {
                  ShowProgressBar(chunk.data_header().receive_time_ns(),
                                  start_time_ns, end_time_ns);
                  MessageService::instance()->Send(
                      static_cast<AdapterConfig::MessageType>(
                          chunk.data_header().message_type()),
                      reinterpret_cast<const unsigned char *>(
                          chunk.message_data().c_str()),
                      chunk.message_data().length());
                });
  player.FinishPlay();
  std::cout << std::endl << " done..." << std::endl;
  exit(0);
}
