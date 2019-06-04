#include "modules/common/message/utils.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/un.h>
#include <string>
#include <vector>
#include <thread>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modules/common/message/proto/message_header.pb.h"

namespace roadstar {
namespace common {
namespace message {

class UtilsTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    std::vector<SockAddr> bind_addr;
    int test_port = FindUnusedLocalTcpPort();
    bind_addr.push_back(SockAddr("/tmp/utilstest.socket"));
    bind_addr.push_back(SockAddr("127.0.0.1:" + std::to_string(test_port)));

    for (unsigned i = 0; i != bind_addr.size(); i++) {
      int fd_bind = socket(bind_addr[i].GetAddressFamily(), SOCK_STREAM, 0);
      fd_bind_.push_back(fd_bind);
      Bind(fd_bind_[i], bind_addr[i].GetRawSockAddr(), bind_addr[i].GetRawSockAddrLength());
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
    for (unsigned i = 0; i != fd_bind_.size(); i++) {
      close(fd_a_[i]);
      close(fd_b_[i]);
      close(fd_bind_[i]);
    }
  }

  std::vector<int> fd_bind_;
  std::vector<int> fd_a_;
  std::vector<int> fd_b_;
};

TEST_F(UtilsTest, SendAndReceiveProto) {
  for (unsigned i = 0; i != fd_bind_.size(); i++) {
    MessageHeader message_header;
    message_header.set_signature("test");
    MessageHeader received;
    received.set_signature("");
    EXPECT_TRUE(SendProto(fd_a_[i], message_header));
    EXPECT_TRUE(RecvProto(fd_b_[i], &received));
    EXPECT_EQ(message_header.signature(), received.signature());
  }
}

TEST_F(UtilsTest, SendAndReceiveBigProto) {
  for (unsigned i = 0; i != fd_bind_.size(); i++) {
    std::string big = "a";
    for (int i = 0; i < (1 << 22); i++) {  // 4M
      big.push_back('a');
    }
    MessageHeader message_header;
    message_header.set_signature(big);
    MessageHeader received;
    received.set_signature("");
    std::thread sender([this, message_header, i]() {
      EXPECT_TRUE(SendProto(fd_a_[i], message_header));
    });
    EXPECT_TRUE(RecvProto(fd_b_[i], &received));
    EXPECT_EQ(message_header.signature(), received.signature());
    sender.join();
  }
}

TEST_F(UtilsTest, SenderClosed) {
  for (unsigned i = 0; i != fd_bind_.size(); i++) {
    close(fd_a_[i]);
    MessageHeader received;
    EXPECT_FALSE(RecvProto(fd_b_[i], &received));
  }
}

TEST_F(UtilsTest, ReceiverClosed) {
  for (unsigned i = 0; i != fd_bind_.size(); i++) {
    close(fd_a_[i]);
    MessageHeader message_header;
    message_header.set_signature("test");
    EXPECT_FALSE(SendProto(fd_b_[i], message_header));
  }
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
