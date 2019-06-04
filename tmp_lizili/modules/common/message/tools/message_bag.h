#ifndef MODULES_COMMON_SERVICE_MESSAGE_TOOLS_MESSAGE_BAG_H
#define MODULES_COMMON_SERVICE_MESSAGE_TOOLS_MESSAGE_BAG_H

#include <limits>
#include <map>
#include <memory>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "modules/common/message/tools/proto/message_bag.pb.h"

namespace roadstar {
namespace common {
namespace message {

// NOT thread-safe
class BagReader {
 public:
  explicit BagReader(const std::vector<std::string>& bags,
      const std::unordered_set<int>& ignore_set = std::unordered_set<int>());
  bool Next(BagDataChunk* chunk);
  void Reset(size_t start_index = 0) {
    current_index_ = start_index;
  }
  void ResetToTimestamp(uint64_t timestamp);

  // Getters
  std::map<int, int> GetTypeCount() {
    return type_count_;
  }
  uint64_t GetStartTimeNs() {
    return start_time_ns_;
  }
  uint64_t GetEndTimeNs() {
    return end_time_ns_;
  }
  std::vector<BagIndex> GetBagIndex();

 private:
  std::vector<std::unique_ptr<std::ifstream>> bag_streams_;
  std::vector<std::pair<BagIndex::BagIndexUnit, std::ifstream*>>
      global_index_;  // (index, #file)
  std::map<int, int> type_count_;
  uint64_t start_time_ns_ = std::numeric_limits<uint64_t>::max();
  uint64_t end_time_ns_ = std::numeric_limits<uint64_t>::min();

  size_t current_index_ = 0;

  void ParseHeader(const std::unordered_set<int>& ignore_set);
};

// NOT thread-safe
class BagWriter {
 public:
  explicit BagWriter(const std::string& filename);
  void FeedData(const BagDataChunk& chunk);
  void Close();
  ~BagWriter() {
    Close();
  }

 private:
  std::unique_ptr<std::ofstream> bag_stream_;
  BagIndex bag_index_;
  uint64_t start_time_ns_ = std::numeric_limits<uint64_t>::max();
  uint64_t end_time_ns_ = std::numeric_limits<uint64_t>::min();
};

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif
