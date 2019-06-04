#ifndef MODULES_COMMON_SERVICE_TOOLS_UTILS_H
#define MODULES_COMMON_SERVICE_TOOLS_UTILS_H

#include <termios.h>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>
#include <thread>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include "modules/common/macro.h"

namespace roadstar {
namespace common {
namespace message {

class Hertz {
 public:
  explicit Hertz(double history_limit_sec) {
    history_limit_ns_ = history_limit_sec * 1000000000ll;
  }
  void ReceiveOneMessage();
  double GetHz();

 private:
  std::mutex mutex_;
  std::queue<uint64_t> msg_queue_;
  uint64_t history_limit_ns_;
};

class KeyboardListener {
 public:
  void StartListen();
  void RegisterKey(char key, std::function<void()> reaction);
  ~KeyboardListener();
  
 private:
  std::atomic<bool> listen_ = false;
  std::unordered_map<unsigned char, std::function<void()>> factory_map_;
  std::unique_ptr<std::thread> listen_keyboard_thread_;
  termios old_term_;

  void ListenKeyboardThread();
  void SetupTerminal();
  void RecoverTerminal();
};


inline double NanoSecondToSecond(uint64_t time_nsec) {
  return static_cast<double>(time_nsec) / 1000000000ll;
}
uint64_t GetTimestampNs();
void SleepUntil(uint64_t eta);
void ParseTypeList(const std::string& type_list,
                   std::unordered_set<int>* type_set);

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif
