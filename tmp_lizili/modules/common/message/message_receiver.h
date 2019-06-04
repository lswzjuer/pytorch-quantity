#ifndef MODULES_COMMON_SERVICE_MESSAGE_RECEIVER_H
#define MODULES_COMMON_SERVICE_MESSAGE_RECEIVER_H

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/message/proto/diagnose.pb.h"

namespace roadstar {
namespace common {
namespace message {

class MessageReceiver {
 public:
  typedef std::function<void(adapter::AdapterConfig::MessageType,
                             const std::vector<unsigned char>&, bool)>
      Callback;
  MessageReceiver(int fd, const Callback& callback);
  virtual ~MessageReceiver();
  void Diagnose(MessageReceiverStatus* status);
  bool IsConnected() const {
    return connected_;
  }

 private:
  int fd_;
  Callback callback_;
  std::atomic<bool> connected_;
  std::unique_ptr<std::thread> thread_;

  // for diagnose
  uint64_t msgs_received_;
  uint64_t bytes_received_;
  std::string remote_name_;
  adapter::AdapterConfig::MessageType message_type_;

  void RunLoop();
  MessageReceiver(const MessageReceiver&) = delete;
  MessageReceiver& operator=(const MessageReceiver&) = delete;
};

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif
