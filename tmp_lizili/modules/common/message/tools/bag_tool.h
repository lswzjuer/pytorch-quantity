#ifndef MODULES_COMMON_SERVICE_MESSAGE_TOOLS_BAG_TOOL_H
#define MODULES_COMMON_SERVICE_MESSAGE_TOOLS_BAG_TOOL_H

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <list>
#include <map>
#include <thread>
#include <functional>
#include <condition_variable>
#include <unordered_set>

#include "modules/common/macro.h"
#include "modules/common/message/tools/utils.h"
#include "modules/common/message/tools/message_bag.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/message/tools/proto/message_bag.pb.h"

namespace roadstar {
namespace common {
namespace message {

class BagPlayer{
 public:
  using OnChunkReady = std::function<void(const BagDataChunk& chunk, 
      uint64_t start_time_ns, uint64_t end_time_ns)>;

  BagPlayer(const std::vector<std::string>& filenames,
            const std::string& ignore_list);
  ~BagPlayer();
  void Launch(double start_time, double duration_time,
              double speed_rate, bool loop, OnChunkReady callback);
  void ShowBagInfo();
  void TogglePaused();
  void StepAhead();
  void FinishPlay(bool force = false);

 private:
  OnChunkReady callback_;
  std::unique_ptr<BagReader> reader_;
  std::mutex mutex_;
  std::condition_variable cond_;
  bool paused_, step_;
  std::atomic<bool> onplay_;
  double speed_rate_;
  uint64_t bag_start_time_, bag_end_time_;
  std::unique_ptr<std::thread> play_thread_;

  uint64_t WaitAndContinue();
  void PlayThread(bool loop);
};

class BagRecorder{
 public:
  explicit BagRecorder(const size_t queue_size);
  void Open(const std::string& filename);
  void Close();
  void FeedData(const BagDataChunk& chunk);
  ~BagRecorder() {
    Close();
  }

 private:
  std::list<BagDataChunk> buffer_queue_;
  size_t queue_size_;
  std::mutex queue_mutex_, writer_mutex_;
  std::condition_variable cond_;
  std::atomic<bool> writable_;
  std::unique_ptr<BagWriter> writer_;
  std::unique_ptr<std::thread> write_thread_;
  
  void WriteThread();
};

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif