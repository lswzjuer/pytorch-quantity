#ifndef MODULES_COMMON_SERVICE_MESSAGE_SENDER_H
#define MODULES_COMMON_SERVICE_MESSAGE_SENDER_H

#include <arpa/inet.h>
#include <semaphore.h>
#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/message/proto/diagnose.pb.h"
#include "modules/common/message/utils.h"

namespace roadstar {
namespace common {
namespace message {

class MessageSender {
 public:
  MessageSender(const adapter::AdapterConfig::MessageType message_type,
                const std::string &endpoint, const std::string &self_name,
                bool header_only, size_t buffer_size = 10);
  virtual ~MessageSender();
  void Send(std::shared_ptr<std::vector<unsigned char>> buffer);
  void Diagnose(MessageSenderStatus *status);

 private:
  void RunLoop();
  void Connect();  // NOT thread-safe

  std::atomic<bool> shutdown_;
  adapter::AdapterConfig::MessageType message_type_;
  std::atomic<MessageSenderStatus::Status> socket_status_;
  int socket_fd_;
  SockAddr remote_addr_;

  sem_t sem_;
  const std::string endpoint_;
  size_t buffer_size_;
  std::mutex queue_mutex_;
  std::list<std::shared_ptr<std::vector<unsigned char>>> buffer_queue_;
  std::unique_ptr<std::thread> run_thread_;

  // for diagnose
  uint64_t msgs_enqueued_;
  uint64_t msgs_sent_;
  uint64_t bytes_sent_;
  uint64_t disconnects_;
  std::string self_name_;

  bool header_only_;

  MessageSender(const MessageSender &) = delete;
  MessageSender &operator=(const MessageSender &) = delete;
};

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif
