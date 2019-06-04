#ifndef MODULES_COMMON_SERVICE_UTILS_H
#define MODULES_COMMON_SERVICE_UTILS_H

#include <arpa/inet.h>
#include <sys/un.h>
#include <string>

#include "google/protobuf/message.h"

namespace roadstar {
namespace common {
namespace message {

class SockAddr {
 public:
  explicit SockAddr(const std::string& endpoint);
  explicit SockAddr(const socklen_t family):family_(family) {
  }
  sa_family_t GetAddressFamily() {
    return family_;
  }
  socklen_t GetRawSockAddrLength() {
    if ( family_ == AF_UNIX ) {
      return sizeof(un_);
    } else {
      return sizeof(in_);
    }
  }
  sockaddr* GetRawSockAddr() {
    if (family_ == AF_UNIX) {
      return reinterpret_cast<sockaddr*>(&un_);
    } else {
      return reinterpret_cast<sockaddr*>(&in_);
    }
  }

 private:
  sa_family_t family_;
  sockaddr_in in_;
  sockaddr_un un_;
};

int Bind(int fd, sockaddr* addr, socklen_t size);

bool SendAll(int fd, const unsigned char* buf, size_t length);

bool RecvAll(int fd, unsigned char* buf, size_t length);

bool SendProto(int fd, const google::protobuf::MessageLite& proto);

bool RecvProto(int fd, google::protobuf::MessageLite* proto);

int FindUnusedLocalTcpPort();

}  // namespace message
}  // namespace common
}  // namespace roadstar

#endif
