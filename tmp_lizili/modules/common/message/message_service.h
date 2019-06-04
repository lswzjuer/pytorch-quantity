#ifndef MODULES_COMMON_SERVICE_MESSAGE_SERVICE_H
#define MODULES_COMMON_SERVICE_MESSAGE_SERVICE_H

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "google/protobuf/message.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/log.h"
#include "modules/common/macro.h"
#include "modules/common/message/message_receiver.h"
#include "modules/common/message/message_sender.h"
#include "modules/common/message/proto/diagnose.pb.h"
#include "ros/include/ros/ros.h"

namespace roadstar {
namespace common {
namespace message {

class MessageService {
 public:
  static void Init(
      const std::string &module_name,
      std::function<void(adapter::AdapterConfig::MessageType,
                         const std::vector<unsigned char> &buffer, bool)>
          callback);

  void InitImpl(
      const std::string &module_name,
      std::function<void(adapter::AdapterConfig::MessageType,
                         const std::vector<unsigned char> &buffer, bool)>
          callback);

  // TODO(wanxinyi): Move seraizliation logic to adapter and move header logic
  // to MessageSender
  template <class T>
  typename std::enable_if<
      std::is_base_of<google::protobuf::MessageLite, T>::value>::type
  Send(adapter::AdapterConfig::MessageType message_type, const T &message) {
    size_t message_len = message.ByteSizeLong();
    std::shared_ptr<std::vector<unsigned char>> buffer(
        new std::vector<unsigned char>(message_len));
    message.SerializeToArray(&(*buffer)[0], message_len);
    SendImpl(message_type, buffer);
  }

  template <class T>
  typename std::enable_if<
      !std::is_base_of<google::protobuf::MessageLite, T>::value>::type
  Send(adapter::AdapterConfig::MessageType message_type, const T &message) {
    size_t message_len = ros::serialization::serializationLength(message);
    std::shared_ptr<std::vector<unsigned char>> buffer(
        new std::vector<unsigned char>(message_len));

    ros::serialization::OStream stream(&((*buffer)[0]), message_len);
    ros::serialization::serialize(stream, message);
    SendImpl(message_type, buffer);
  }

  void Send(adapter::AdapterConfig::MessageType message_type,
            const unsigned char *data, size_t len) {
    std::shared_ptr<std::vector<unsigned char>> buffer(
        new std::vector<unsigned char>(len));
    memcpy(&(*buffer)[0], data, len);
    SendImpl(message_type, buffer);
  }

  void Diagnose(MessageServiceStatus *status);
  void ShutDown();

  DECLARE_SINGLETON(MessageService);

  ~MessageService();

 private:
  bool initialized_ = false;
  std::atomic<bool> shutdown_;
  std::string module_name_;
  // message_type -> [{module, endpoint_sender}]
  std::unordered_multimap<
      int, std::pair<std::string, std::unique_ptr<MessageSender>>>
      topic_endpoint_map_;
  std::unordered_set<int> publishing_types_;
  std::string endpoint_;
  std::unique_ptr<std::thread> accept_thread_;
  std::unique_ptr<std::thread> diagnose_thread_;
  std::list<std::unique_ptr<MessageReceiver>> message_receivers_;
  std::function<void(adapter::AdapterConfig::MessageType,
                     const std::vector<unsigned char> &buffer, bool)>
      callback_;

  std::mutex receivers_mutex_;
  int accept_fd_;

  void SendImpl(adapter::AdapterConfig::MessageType message_type,
                std::shared_ptr<std::vector<unsigned char>> buffer);
  void AcceptThread();
  void DiagnoseThread();

  friend class MessageServiceTest_MessageServiceMultipleModules_Test;
  friend class MessageServiceTest_MessageServiceSelfCommunication_Test;
};

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif
