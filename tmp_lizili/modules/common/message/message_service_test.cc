#include "modules/common/message/message_service.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <time.h>
#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include "gflags/gflags.h"
#include "gmock/gmock.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "modules/common/message/proto/message_header.pb.h"
#include "modules/common/message/utils.h"
#include "modules/common/util/file.h"

DECLARE_string(living_modules_path);
DECLARE_string(internal_living_modules_path);
DECLARE_string(module_conf_path);

namespace roadstar {
namespace common {
namespace message {
namespace {
bool CheckStatus(MessageServiceStatus* status, std::string expected_fn) {
  MessageServiceStatus expected_status;
  if (!util::GetProtoFromASCIIFile(expected_fn, &expected_status)) {
    return false;
  }
  status->clear_timestamp_ns();
  for (auto& sender : *status->mutable_senders()) {
    sender.clear_disconnects();
    sender.set_bytes_sent(sender.bytes_sent() > 0);
    if (sender.message_type() == adapter::AdapterConfig::MESSAGE_SERVICE_STATUS) {
      sender.set_msgs_sent(0);
      sender.set_bytes_sent(0);
      sender.set_msgs_enqueued(0);
      sender.set_queue_size(0);
    }
  }
  for (auto& recver : *status->mutable_receivers()) {
    recver.set_bytes_received(recver.bytes_received() > 0);
  }
  std::sort(status->mutable_receivers()->begin(),
            status->mutable_receivers()->end(),
            [](const MessageReceiverStatus& a, const MessageReceiverStatus& b) {
              return a.remote_name() == b.remote_name()
                         ? a.message_type() < b.message_type()
                         : a.remote_name() < b.remote_name();
            });

  std::sort(status->mutable_senders()->begin(),
            status->mutable_senders()->end(),
            [](const MessageSenderStatus& a, const MessageSenderStatus& b) {
              return a.target_module() == b.target_module()
                         ? a.message_type() < b.message_type()
                         : a.target_module() < b.target_module();
            });

  std::sort(expected_status.mutable_senders()->begin(),
            expected_status.mutable_senders()->end(),
            [](const MessageSenderStatus& a, const MessageSenderStatus& b) {
              return a.target_module() == b.target_module()
                         ? a.message_type() < b.message_type()
                         : a.target_module() < b.target_module();
            });
  return google::protobuf::util::MessageDifferencer::Equals(expected_status,
                                                            *status);
}

bool WaitUntil(int tolerate, std::function<bool()> Watcher) {
  time_t start_time = time(NULL);
  while (!Watcher()) {
    time_t current_time = time(NULL);
    if (current_time - start_time > tolerate) {
      return false;
    }
    usleep(10000);
  }
  return true;
}
}  // namespace

class MessageServiceTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    std::vector<SockAddr> bind_addr;
    endpoint_.push_back("/tmp/messageservicetest.socket");
    int test_port = FindUnusedLocalTcpPort();
    endpoint_.push_back("127.0.0.1:" + std::to_string(test_port));

    for (unsigned i = 0; i != endpoint_.size(); i++) {
      bind_addr.push_back(SockAddr(endpoint_[i]));
      int fd_bind = socket(bind_addr[i].GetAddressFamily(), SOCK_STREAM, 0);
      fd_bind_.push_back(fd_bind);
      Bind(fd_bind_[i], bind_addr[i].GetRawSockAddr(),
           bind_addr[i].GetRawSockAddrLength());
      listen(fd_bind_[i], 128);
      std::thread acceptor([this, &bind_addr, i]() {
        SockAddr peer(bind_addr[i].GetAddressFamily());
        socklen_t peer_len = peer.GetRawSockAddrLength();
        int fd_a = accept(fd_bind_[i], peer.GetRawSockAddr(), &peer_len);
        fd_a_.push_back(fd_a);
      });
      int fd_b = socket(bind_addr[i].GetAddressFamily(), SOCK_STREAM, 0);
      fd_b_.push_back(fd_b);
      ASSERT_EQ(connect(fd_b_[i], bind_addr[i].GetRawSockAddr(),
                        bind_addr[i].GetRawSockAddrLength()),
                0);
      acceptor.join();
    }
  }
  virtual void TearDown() {
    for (unsigned i = 0; i != endpoint_.size(); i++) {
      close(fd_a_[i]);
      close(fd_b_[i]);
      close(fd_bind_[i]);
    }
  }
  std::vector<int> fd_bind_;
  std::vector<int> fd_a_;
  std::vector<int> fd_b_;
  std::vector<std::string> endpoint_;
};

TEST_F(MessageServiceTest, CorruptedReceiverAutoExit) {
  for (unsigned i = 0; i != endpoint_.size(); i++) {
    MessageReceiver receiver(fd_a_[i], MessageReceiver::Callback());
    std::vector<size_t> buf = {1, 2, 3, 4};
    SendAll(fd_b_[i], reinterpret_cast<unsigned char*>(&buf[0]), sizeof(buf));
    // Wait for receiver to be unconnected because of remote shutdown;
    EXPECT_TRUE(
        WaitUntil(5, [&receiver]() { return !receiver.IsConnected(); }));
  }
}

TEST_F(MessageServiceTest, ReceiverCallback) {
  for (unsigned i = 0; i != endpoint_.size(); i++) {
    std::atomic<bool> called(false);
    std::vector<unsigned char> data_sent = {'f', 'a', 'b', 'u'};
    MessageReceiver receiver(
        fd_a_[i],
        [&called, &data_sent](adapter::AdapterConfig::MessageType type,
                              const std::vector<unsigned char>& data,
                              bool header_only) {
          EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
          ASSERT_EQ(data, data_sent);
          called = true;
        });
    MessageHeader message_header;
    message_header.set_signature("MESSAGE");
    message_header.set_message_length(data_sent.size());
    message_header.set_sender("test");
    message_header.set_message_type(adapter::AdapterConfig::FUSION_MAP);
    message_header.set_header_only(false);
    SendProto(fd_b_[i], message_header);
    SendAll(fd_b_[i], &data_sent[0], data_sent.size());
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
    EXPECT_TRUE(receiver.IsConnected());
    MessageReceiverStatus status;
    receiver.Diagnose(&status);
    EXPECT_TRUE(status.has_received());
    EXPECT_EQ(status.message_type(), adapter::AdapterConfig::FUSION_MAP);
    EXPECT_EQ(status.msgs_received(), 1);
    EXPECT_GT(status.bytes_received(), data_sent.size());
    EXPECT_EQ(status.remote_name(), message_header.sender());
    close(fd_b_[i]);
    ADEBUG << "closed";
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
  }
}

TEST_F(MessageServiceTest, ReceiverHeaderOnlyCallback) {
  for (unsigned i = 0; i != endpoint_.size(); i++) {
    std::atomic<bool> called(false);
    std::vector<unsigned char> data_empty = {};
    MessageReceiver receiver(
        fd_a_[i],
        [&called, &data_empty](adapter::AdapterConfig::MessageType type,
                               const std::vector<unsigned char>& data,
                               bool header_only) {
          EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
          ASSERT_EQ(data, data_empty);
          ASSERT_EQ(header_only, true);
          called = true;
        });
    MessageHeader message_header;
    message_header.set_signature("MESSAGE");
    message_header.set_message_length(data_empty.size());
    message_header.set_sender("test");
    message_header.set_message_type(adapter::AdapterConfig::FUSION_MAP);
    message_header.set_header_only(true);
    SendProto(fd_b_[i], message_header);
    SendAll(fd_b_[i], &data_empty[0], data_empty.size());
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
    EXPECT_TRUE(receiver.IsConnected());
    MessageReceiverStatus status;
    receiver.Diagnose(&status);
    EXPECT_TRUE(status.has_received());
    EXPECT_EQ(status.message_type(), adapter::AdapterConfig::FUSION_MAP);
    EXPECT_EQ(status.msgs_received(), 1);
    EXPECT_GT(status.bytes_received(), data_empty.size());
    EXPECT_EQ(status.remote_name(), message_header.sender());
    close(fd_b_[i]);
    ADEBUG << "closed";
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
  }
}

TEST_F(MessageServiceTest, SenderWithReceiverHeaderOnly) {
  for (unsigned i = 0; i != endpoint_.size(); i++) {
    std::atomic<bool> called(false);
    std::vector<unsigned char> data_sent = {};
    std::unique_ptr<MessageSender> sender(new MessageSender(
        adapter::AdapterConfig::FUSION_MAP, endpoint_[i], "sender", true));
    // This message should be ingored
    sender->Send(std::make_shared<std::vector<unsigned char>>(data_sent));
    MessageSenderStatus sender_status;
    sender->Diagnose(&sender_status);
    EXPECT_EQ(sender_status.msgs_enqueued(), 1);
    EXPECT_EQ(sender_status.msgs_sent(), 0);

    sa_family_t family = (i == 0) ? AF_UNIX : AF_INET;
    SockAddr peer(family);
    socklen_t peer_len = peer.GetRawSockAddrLength();
    fd_a_[i] = accept(fd_bind_[i], peer.GetRawSockAddr(), &peer_len);
    std::unique_ptr<MessageReceiver> receiver(new MessageReceiver(
        fd_a_[i],
        [&called, &data_sent](adapter::AdapterConfig::MessageType type,
                              const std::vector<unsigned char>& data,
                              bool header_only) {
          called = true;
          EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
          ASSERT_EQ(data, data_sent);
          ASSERT_EQ(header_only, true);
        }));

    EXPECT_TRUE(WaitUntil(5, [&sender, &sender_status]() {
      sender->Diagnose(&sender_status);
      return sender_status.status() == MessageSenderStatus::CONNECTED;
    }));

    sender->Send(std::make_shared<std::vector<unsigned char>>(data_sent));
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
    EXPECT_TRUE(receiver->IsConnected());
    EXPECT_TRUE(WaitUntil(5, [&sender, &sender_status]() {
      sender->Diagnose(&sender_status);
      return sender_status.msgs_sent() == 1;
    }));

    MessageReceiverStatus status;
    receiver->Diagnose(&status);
    EXPECT_TRUE(status.has_received());
    EXPECT_EQ(status.message_type(), adapter::AdapterConfig::FUSION_MAP);
    EXPECT_EQ(status.msgs_received(), 1);
    EXPECT_GT(status.bytes_received(), data_sent.size());
    EXPECT_EQ(status.remote_name(), "sender");

    sender->Diagnose(&sender_status);
    EXPECT_EQ(sender_status.msgs_enqueued(), 2);
    EXPECT_EQ(sender_status.msgs_sent(), 1);
    EXPECT_EQ(sender_status.status(), MessageSenderStatus::CONNECTED);

    // Shutdown receiver, sender will reconnect
    receiver.reset();
    sender->Send(std::make_shared<std::vector<unsigned char>>(
        data_sent));  // this will also be ignored because of broken connection

    // Receiver online again
    called = false;
    fd_a_[i] = accept(fd_bind_[i], peer.GetRawSockAddr(), &peer_len);
    receiver.reset(new MessageReceiver(
        fd_a_[i],
        [&called, &data_sent](adapter::AdapterConfig::MessageType type,
                              const std::vector<unsigned char>& data,
                              bool header_only) {
          called = true;
          EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
          ASSERT_EQ(data, data_sent);
        }));
    EXPECT_TRUE(WaitUntil(5, [&sender, &sender_status]() {
      sender->Diagnose(&sender_status);
      return sender_status.status() == MessageSenderStatus::CONNECTED;
    }));
    sender->Send(std::make_shared<std::vector<unsigned char>>(data_sent));
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
    sender->Diagnose(&sender_status);
    EXPECT_EQ(sender_status.msgs_enqueued(), 4);
    EXPECT_EQ(sender_status.msgs_sent(), 2);
    EXPECT_EQ(sender_status.status(), MessageSenderStatus::CONNECTED);

    sender.reset();
    EXPECT_TRUE(
        WaitUntil(5, [&receiver]() { return !receiver->IsConnected(); }));
    receiver.reset();
  }
}

TEST_F(MessageServiceTest, SenderWithReceiver) {
  for (unsigned i = 0; i != endpoint_.size(); i++) {
    std::atomic<bool> called(false);
    std::vector<unsigned char> data_sent = {'f', 'a', 'b', 'u'};
    std::unique_ptr<MessageSender> sender(new MessageSender(
        adapter::AdapterConfig::FUSION_MAP, endpoint_[i], "sender", false));
    // This message should be ingored
    sender->Send(std::make_shared<std::vector<unsigned char>>(data_sent));
    MessageSenderStatus sender_status;
    sender->Diagnose(&sender_status);
    EXPECT_EQ(sender_status.msgs_enqueued(), 1);
    EXPECT_EQ(sender_status.msgs_sent(), 0);

    sa_family_t family = (i == 0) ? AF_UNIX : AF_INET;
    SockAddr peer(family);
    socklen_t peer_len = peer.GetRawSockAddrLength();
    fd_a_[i] = accept(fd_bind_[i], peer.GetRawSockAddr(), &peer_len);
    std::unique_ptr<MessageReceiver> receiver(new MessageReceiver(
        fd_a_[i],
        [&called, &data_sent](adapter::AdapterConfig::MessageType type,
                              const std::vector<unsigned char>& data,
                              bool header_only) {
          called = true;
          EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
          ASSERT_EQ(data, data_sent);
          ASSERT_EQ(header_only, false);
        }));

    EXPECT_TRUE(WaitUntil(5, [&sender, &sender_status]() {
      sender->Diagnose(&sender_status);
      return sender_status.status() == MessageSenderStatus::CONNECTED;
    }));

    sender->Send(std::make_shared<std::vector<unsigned char>>(data_sent));
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
    EXPECT_TRUE(receiver->IsConnected());
    EXPECT_TRUE(WaitUntil(5, [&sender, &sender_status]() {
      sender->Diagnose(&sender_status);
      return sender_status.msgs_sent() == 1;
    }));

    MessageReceiverStatus status;
    receiver->Diagnose(&status);
    EXPECT_TRUE(status.has_received());
    EXPECT_EQ(status.message_type(), adapter::AdapterConfig::FUSION_MAP);
    EXPECT_EQ(status.msgs_received(), 1);
    EXPECT_GT(status.bytes_received(), data_sent.size());
    EXPECT_EQ(status.remote_name(), "sender");

    sender->Diagnose(&sender_status);
    EXPECT_EQ(sender_status.msgs_enqueued(), 2);
    EXPECT_EQ(sender_status.msgs_sent(), 1);
    EXPECT_EQ(sender_status.status(), MessageSenderStatus::CONNECTED);

    // Shutdown receiver, sender will reconnect
    receiver.reset();
    sender->Send(std::make_shared<std::vector<unsigned char>>(
        data_sent));  // this will also be ignored because of broken connection

    // Receiver online again
    called = false;
    fd_a_[i] = accept(fd_bind_[i], peer.GetRawSockAddr(), &peer_len);
    receiver.reset(new MessageReceiver(
        fd_a_[i],
        [&called, &data_sent](adapter::AdapterConfig::MessageType type,
                              const std::vector<unsigned char>& data,
                              bool header_only) {
          called = true;
          EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
          ASSERT_EQ(data, data_sent);
        }));
    EXPECT_TRUE(WaitUntil(5, [&sender, &sender_status]() {
      sender->Diagnose(&sender_status);
      return sender_status.status() == MessageSenderStatus::CONNECTED;
    }));
    sender->Send(std::make_shared<std::vector<unsigned char>>(data_sent));
    EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));
    sender->Diagnose(&sender_status);
    EXPECT_EQ(sender_status.msgs_enqueued(), 4);
    EXPECT_EQ(sender_status.msgs_sent(), 2);
    EXPECT_EQ(sender_status.status(), MessageSenderStatus::CONNECTED);

    sender.reset();
    EXPECT_TRUE(
        WaitUntil(5, [&receiver]() { return !receiver->IsConnected(); }));
    receiver.reset();
  }
}

TEST_F(MessageServiceTest, MessageServiceSelfCommunication) {
  FLAGS_living_modules_path =
      "modules/common/message/testdata/living_modules.pb.txt";
  FLAGS_internal_living_modules_path =
      "modules/common/message/testdata/internal_living_modules.pb.txt";
  FLAGS_module_conf_path = "modules/common/message/testdata/module_conf.pb.txt";
  std::vector<unsigned char> data_sent = {'f', 'a', 'b', 'u'};
  std::atomic<bool> called(false);
  MessageService service;
  service.InitImpl("module_a", [&called, &data_sent](
                                   adapter::AdapterConfig::MessageType type,
                                   const std::vector<unsigned char>& buffer,
                                   bool header_only) {
    EXPECT_EQ(type, adapter::AdapterConfig::FUSION_MAP);
    EXPECT_EQ(data_sent, buffer);
    called = true;
  });

  EXPECT_TRUE(WaitUntil(5, [&service]() {
    MessageServiceStatus status;
    service.Diagnose(&status);
    return std::any_of(status.senders().begin(), status.senders().end(),
                       [](const MessageSenderStatus sender_status) {
                         return sender_status.target_module() == "module_a" &&
                                sender_status.status() ==
                                    MessageSenderStatus::CONNECTED;
                       });
  }));

  service.Send(adapter::AdapterConfig::FUSION_MAP, &data_sent[0],
               data_sent.size());
  EXPECT_TRUE(WaitUntil(5, [&called]() { return called == true; }));

  EXPECT_TRUE(WaitUntil(10, [&service]() {
    MessageServiceStatus status;
    service.Diagnose(&status);
    return CheckStatus(
        &status,
        "modules/common/message/testdata/module_a_status_with_ab.pb.txt");
  }));
}

TEST_F(MessageServiceTest, MessageServiceMultipleModules) {
  FLAGS_living_modules_path =
      "modules/common/message/testdata/living_modules.pb.txt";
  FLAGS_internal_living_modules_path =
      "modules/common/message/testdata/internal_living_modules.pb.txt";
  FLAGS_module_conf_path = "modules/common/message/testdata/module_conf.pb.txt";
  std::mutex data_lock;
  MessageService a;
  std::vector<std::pair<adapter::AdapterConfig::MessageType, char>> data_a;
  a.InitImpl("module_a",
             [&data_lock, &data_a](adapter::AdapterConfig::MessageType type,
                                   const std::vector<unsigned char>& buffer,
                                   bool header_only) {
               EXPECT_EQ(buffer.size(), 1);
               std::lock_guard<std::mutex> lock(data_lock);
               data_a.push_back(std::make_pair(type, buffer[0]));
             });
  // b is missing
  MessageService c;
  std::vector<std::pair<adapter::AdapterConfig::MessageType, char>> data_c;
  c.InitImpl("module_c",
             [&data_lock, &data_c](adapter::AdapterConfig::MessageType type,
                                   const std::vector<unsigned char>& buffer,
                                   bool header_only) {
               EXPECT_EQ(buffer.size(), 1);
               std::lock_guard<std::mutex> lock(data_lock);
               data_c.push_back(std::make_pair(type, buffer[0]));
             });
  MessageService d;
  std::vector<std::pair<adapter::AdapterConfig::MessageType, char>> data_d;
  d.InitImpl("module_d",
             [&data_lock, &data_d](adapter::AdapterConfig::MessageType type,
                                   const std::vector<unsigned char>& buffer,
                                   bool header_only) {
               EXPECT_EQ(buffer.size(), 1);
               std::lock_guard<std::mutex> lock(data_lock);
               data_d.push_back(std::make_pair(type, buffer[0]));
             });
  MessageService e;  // message service with RECEIVE_HEADER mode
  std::vector<std::pair<adapter::AdapterConfig::MessageType, char>> data_e;
  e.InitImpl("module_e", [&data_lock, &data_e](
                             adapter::AdapterConfig::MessageType type,
                             const std::vector<unsigned char>& buffer,
                             bool header_only) {
    std::lock_guard<std::mutex> lock(data_lock);
    EXPECT_EQ(header_only, buffer.size() == 0);
    data_e.push_back(std::make_pair(type, header_only ? '@' : buffer[0]));
  });
  MessageService f;
  std::vector<std::pair<adapter::AdapterConfig::MessageType, char>> data_f;
  f.InitImpl("module_f",
             [&data_lock, &data_f](adapter::AdapterConfig::MessageType type,
                                   const std::vector<unsigned char>& buffer,
                                   bool header_only) {
               if (type == adapter::AdapterConfig::MESSAGE_SERVICE_STATUS) {
                 return;
               }
               std::lock_guard<std::mutex> lock(data_lock);
               data_f.push_back(std::make_pair(type, buffer[0]));
             });
  EXPECT_TRUE(WaitUntil(10, [&a]() {
    MessageServiceStatus status;
    a.Diagnose(&status);
    return CheckStatus(&status,
                       "modules/common/message/testdata/"
                       "module_a_init_status_with_abcd.pb.txt");
  }));

  EXPECT_TRUE(WaitUntil(10, [&c]() {
    MessageServiceStatus status;
    c.Diagnose(&status);
    return CheckStatus(&status,
                       "modules/common/message/testdata/"
                       "module_c_init_status_with_abcd.pb.txt");
  }));

  EXPECT_TRUE(WaitUntil(10, [&d]() {
    MessageServiceStatus status;
    d.Diagnose(&status);
    return std::all_of(status.senders().begin(), status.senders().end(),
                       [](const MessageSenderStatus sender_status) {
                         return sender_status.target_module() == "module_b" ||
                                sender_status.status() ==
                                    MessageSenderStatus::CONNECTED;
                       });
  }));

  a.Send(adapter::AdapterConfig::FUSION_MAP,
         reinterpret_cast<const unsigned char*>("1"), 1);
  a.Send(adapter::AdapterConfig::FUSION_MAP,
         reinterpret_cast<const unsigned char*>("2"), 1);
  a.Send(adapter::AdapterConfig::PLANNING_TRAJECTORY,
         reinterpret_cast<const unsigned char*>("3"), 1);
  c.Send(adapter::AdapterConfig::FUSION_MAP,
         reinterpret_cast<const unsigned char*>("4"), 1);
  c.Send(adapter::AdapterConfig::FUSION_MAP,
         reinterpret_cast<const unsigned char*>("5"), 1);
  c.Send(adapter::AdapterConfig::LOCALIZATION,
         reinterpret_cast<const unsigned char*>("6"), 1);
  d.Send(adapter::AdapterConfig::CONTROL_COMMAND,
         reinterpret_cast<const unsigned char*>("7"), 1);
  d.Send(adapter::AdapterConfig::FUSION_MAP,
         reinterpret_cast<const unsigned char*>("8"), 1);

  EXPECT_TRUE(WaitUntil(10, [&data_lock, &data_a, &data_c, &data_f]() {
    std::lock_guard<std::mutex> lock(data_lock);
    size_t data_size = data_a.size() + data_c.size() + data_f.size();
    return data_size == 20;
  }));

  EXPECT_TRUE(WaitUntil(10, [&a]() {
    MessageServiceStatus status;
    a.Diagnose(&status);
    return CheckStatus(
        &status,
        "modules/common/message/testdata/module_a_status_with_abcd.pb.txt");
  }));

  EXPECT_TRUE(WaitUntil(10, [&c]() {
    MessageServiceStatus status;
    c.Diagnose(&status);
    return CheckStatus(
        &status,
        "modules/common/message/testdata/module_c_status_with_abcd.pb.txt");
  }));

  EXPECT_TRUE(WaitUntil(5, [&d]() {
    MessageServiceStatus status;
    d.Diagnose(&status);
    return status.senders_size() > 3;
  }));

  std::sort(data_a.begin(), data_a.end());
  EXPECT_EQ(data_a, decltype(data_a)({
                        {adapter::AdapterConfig::LOCALIZATION, '6'},
                        {adapter::AdapterConfig::FUSION_MAP, '1'},
                        {adapter::AdapterConfig::FUSION_MAP, '2'},
                        {adapter::AdapterConfig::FUSION_MAP, '4'},
                        {adapter::AdapterConfig::FUSION_MAP, '5'},
                        {adapter::AdapterConfig::FUSION_MAP, '8'},
                    }));
  std::sort(data_c.begin(), data_c.end());
  EXPECT_EQ(data_c, decltype(data_c)({
                        {adapter::AdapterConfig::FUSION_MAP, '1'},
                        {adapter::AdapterConfig::FUSION_MAP, '2'},
                        {adapter::AdapterConfig::FUSION_MAP, '4'},
                        {adapter::AdapterConfig::FUSION_MAP, '5'},
                        {adapter::AdapterConfig::FUSION_MAP, '8'},
                        {adapter::AdapterConfig::PLANNING_TRAJECTORY, '3'},
                    }));
  EXPECT_TRUE(data_d.empty());
  std::sort(data_e.begin(), data_e.end());
  EXPECT_EQ(data_e, decltype(data_e)({
                        {adapter::AdapterConfig::LOCALIZATION, '6'},
                        {adapter::AdapterConfig::PLANNING_TRAJECTORY, '@'},
                    }));
  std::sort(data_e.begin(), data_e.end());
  std::sort(data_f.begin(), data_f.end());
  EXPECT_EQ(data_f, decltype(data_f)({
                        {adapter::AdapterConfig::CONTROL_COMMAND, '7'},
                        {adapter::AdapterConfig::LOCALIZATION, '6'},
                        {adapter::AdapterConfig::FUSION_MAP, '1'},
                        {adapter::AdapterConfig::FUSION_MAP, '2'},
                        {adapter::AdapterConfig::FUSION_MAP, '4'},
                        {adapter::AdapterConfig::FUSION_MAP, '5'},
                        {adapter::AdapterConfig::FUSION_MAP, '8'},
                        {adapter::AdapterConfig::PLANNING_TRAJECTORY, '3'},
                    }));
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
