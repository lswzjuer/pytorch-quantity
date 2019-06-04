#include "modules/common/message/utils.h"

#include <vector>

#include "modules/common/log.h"
#include "unistd.h"

namespace roadstar {
namespace common {
namespace message {
namespace {

sa_family_t EndpointType(const std::string &endpoint) {
  auto colon = endpoint.find(':');
  if (colon == std::string::npos || colon == endpoint.length() - 1) {
    return AF_UNIX;
  } else {
    return AF_INET;
  }
}

void EndpointStrToAddr(const std::string &endpoint, sockaddr *addr) {
  auto colon = endpoint.find(':');
  if (colon == std::string::npos || colon == endpoint.length() - 1) {
    auto addr_un = reinterpret_cast<sockaddr_un *>(addr);
    memset(addr_un, 0, sizeof(*addr_un));
    addr_un->sun_family = AF_UNIX;
    if (endpoint.size() > sizeof(addr_un->sun_path) - 1) {
      AFATAL << "Invalid addr " << endpoint;
    }
    strncpy(addr_un->sun_path, endpoint.c_str(), sizeof(addr_un->sun_path) - 1);
  } else {
    auto addr_in = reinterpret_cast<sockaddr_in *>(addr);
    int port = atoi(endpoint.substr(colon + 1).c_str());
    addr_in->sin_family = AF_INET;
    addr_in->sin_port = htons(port);
    if (inet_pton(AF_INET, endpoint.substr(0, colon).c_str(),
                  &addr_in->sin_addr) != 1) {
      AFATAL << "Invalid addr " << endpoint;
    }
  }
}

}  // namespace

SockAddr::SockAddr(const std::string &endpoint) {
  family_ = EndpointType(endpoint);
  EndpointStrToAddr(endpoint, GetRawSockAddr());
}

int Bind(int fd, sockaddr *addr, socklen_t size) {
  if (addr->sa_family == AF_UNIX) {
    unlink(addr->sa_data);
  }
  return bind(fd, addr, size);
}

bool SendAll(int fd, const unsigned char *buf, size_t length) {
  size_t bytes_sent = 0;
  while (bytes_sent < length) {
    ssize_t b = send(fd, buf + bytes_sent, length - bytes_sent, MSG_NOSIGNAL);
    if (b > 0) {
      bytes_sent += b;
    } else {
      return false;
    }
  }
  return true;
}

bool RecvAll(int fd, unsigned char *buf, size_t length) {
  size_t left = length;
  while (left > 0) {
    ssize_t received = recv(fd, buf + (length - left), left, 0);
    if (received <= 0) {
      return false;
    }
    left -= received;
  }
  return true;
}

bool SendProto(int fd, const google::protobuf::MessageLite &proto) {
  std::vector<unsigned char> buffer(proto.ByteSizeLong());
  uint64_t buffer_length = htole64(buffer.size());
  proto.SerializeToArray(&buffer[0], buffer.size());
  return SendAll(fd, reinterpret_cast<unsigned char *>(&buffer_length),
                 sizeof(buffer_length)) &&
         SendAll(fd, &buffer[0], buffer.size());
}

bool RecvProto(int fd, google::protobuf::MessageLite *proto) {
  uint64_t buffer_length;
  if (!RecvAll(fd, reinterpret_cast<unsigned char *>(&buffer_length),
               sizeof(buffer_length))) {
    return false;
  }
  buffer_length = le64toh(buffer_length);
  if (buffer_length > (1ull << 32)) {
    return false;
  }
  std::vector<unsigned char> buffer(buffer_length);
  return RecvAll(fd, &buffer[0], buffer_length) &&
         proto->ParseFromArray(&buffer[0], buffer_length);
}

int FindUnusedLocalTcpPort() {
  int sfd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

  int port = 12345;
  const int port_limit = 60000;
  addr.sin_port = htons(port);
  while (port < port_limit &&
         bind(sfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    port++;
    addr.sin_port = htons(port);
  }
  close(sfd);

  if (port >= port_limit) {
    return -1;
  }
  return port;
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
