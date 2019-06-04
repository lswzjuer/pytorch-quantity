#include "modules/common/message/tools/utils.h"

#include <time.h>
#include <unistd.h>
#include <functional>

#include "modules/common/log.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"

namespace roadstar {
namespace common {
namespace message {
namespace {

}  // namespace

uint64_t GetTimestampNs() {
  timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  return now.tv_sec * 1000000000ll + now.tv_nsec;
}

void SleepUntil(uint64_t eta) {
  while (true) {
    uint64_t now = GetTimestampNs();
    if (now >= eta) {
      break;
    }
    timespec left;
    left.tv_sec = (eta - now) / 1000000000ll;
    left.tv_nsec = (eta - now) % 1000000000ll;
    clock_nanosleep(CLOCK_REALTIME, 0, &left, NULL);
  }
}

void ParseTypeList(const std::string& type_list,
                  std::unordered_set<int>* type_set) {
  if (type_list.empty()) {
    return;
  }
  size_t pos_left = 0, pos_right = 0;
  adapter::AdapterConfig::MessageType message_type;
  while (pos_left < type_list.length()) {
    pos_right = type_list.find(',', pos_left);
    if (pos_right == std::string::npos) {
      pos_right = type_list.length();
    }
    const std::string& type_str =
          type_list.substr(pos_left, pos_right-pos_left);
    if (!adapter::AdapterConfig::MessageType_Parse(type_str, &message_type)) {
      AFATAL << "Undefined message type " << type_str;
    } else {
      type_set->insert(message_type);
      AINFO << "Message type "
            << adapter::AdapterConfig::MessageType_Name(message_type);
    }
    pos_left = pos_right+1;
  }
}

void Hertz::ReceiveOneMessage() {
  std::lock_guard<std::mutex> lock(mutex_);
  msg_queue_.push(GetTimestampNs());
}

double Hertz::GetHz() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (msg_queue_.empty()) {
    return 0;
  }
  uint64_t now = GetTimestampNs();
  while (!msg_queue_.empty() && now - msg_queue_.front() > history_limit_ns_) {
    msg_queue_.pop();
  }
  return static_cast<double>(msg_queue_.size()) /
        NanoSecondToSecond(history_limit_ns_);
}

KeyboardListener::~KeyboardListener() {
  listen_ = false;
  RecoverTerminal();
  listen_keyboard_thread_->join();
  listen_keyboard_thread_.reset();
}
void KeyboardListener::StartListen() {
  listen_ = true;
  SetupTerminal();
  listen_keyboard_thread_.reset(new std::thread(
      std::bind(&KeyboardListener::ListenKeyboardThread, this)));
}

void KeyboardListener::RegisterKey(
    char key, std::function<void()> reaction) {
  if (listen_) {
    return;
  }
  factory_map_[key] = reaction;
}

void KeyboardListener::SetupTerminal() {
  tcgetattr(0, &old_term_);
  termios new_term = old_term_;
  new_term.c_lflag &= ~ICANON;
  new_term.c_lflag &= ~ECHO;
  new_term.c_cc[VMIN]  = 1;
  new_term.c_cc[VTIME] = 0;
  tcsetattr(0, TCSANOW, &new_term);
}

void KeyboardListener::RecoverTerminal(){
  tcsetattr(0, TCSANOW, &old_term_);
}

void KeyboardListener::ListenKeyboardThread() {
  while(listen_) {
    auto key = getchar();
    if (!factory_map_.count(key)) {
      continue;
    }
    factory_map_[key]();
  }
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
