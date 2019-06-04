#include "modules/common/message/message_sender.h"

#include <arpa/inet.h>
#include <endian.h>
#include <netinet/tcp.h>
#include <semaphore.h>
#include <signal.h>
#include <unistd.h>
#include <cstring>

#include "modules/common/log.h"
#include "modules/common/message/proto/message_header.pb.h"
#include "modules/common/message/tools/utils.h"
#include "modules/common/message/utils.h"

namespace roadstar {
namespace common {
namespace message {
namespace {
std::once_flag once_flag;
static const int kMessageSenderWakeSignal = SIGRTMIN + 1;
void SigHandler(int sig) {
  // Do nothing
}
void RegisterSigHandler() {
  AFATAL_IF(signal(kMessageSenderWakeSignal, &SigHandler) != nullptr)
      << "Conflicting real-time signal: " << kMessageSenderWakeSignal;
}
}  // namespace

MessageSender::MessageSender(
    const adapter::AdapterConfig::MessageType message_type,
    const std::string &endpoint, const std::string &self_name, bool header_only,
    size_t buffer_size)
    : shutdown_(false),
      message_type_(message_type),
      socket_status_(MessageSenderStatus::NEW),
      socket_fd_(-1),
      remote_addr_(endpoint),
      endpoint_(endpoint),
      buffer_size_(buffer_size),
      msgs_enqueued_(0),
      msgs_sent_(0),
      bytes_sent_(0),
      disconnects_(0),
      self_name_(self_name),
      header_only_(header_only) {
  sem_init(&sem_, 0, 0);
  std::call_once(once_flag, RegisterSigHandler);

  run_thread_.reset(new std::thread(&MessageSender::RunLoop, this));
}

MessageSender::~MessageSender() {
  shutdown_ = true;
  sem_post(&sem_);
  shutdown(socket_fd_, SHUT_RDWR);
  close(socket_fd_);
  // Wake threads
  pthread_kill(run_thread_->native_handle(), kMessageSenderWakeSignal);
  run_thread_->join();
}

void MessageSender::Send(std::shared_ptr<std::vector<unsigned char>> buffer) {
  if (shutdown_) {
    return;
  }
  std::lock_guard<std::mutex> guard(queue_mutex_);
  buffer_queue_.push_back(buffer);
  msgs_enqueued_++;
  while (buffer_queue_.size() > buffer_size_) {
    buffer_queue_.pop_front();
  }
  sem_post(&sem_);
}

void MessageSender::RunLoop() {
  while (!shutdown_) {
    Connect();

    sem_wait(&sem_);
    queue_mutex_.lock();
    if (buffer_queue_.empty()) {
      queue_mutex_.unlock();
      continue;
    }
    std::shared_ptr<std::vector<unsigned char>> msg = buffer_queue_.front();
    buffer_queue_.pop_front();
    queue_mutex_.unlock();

    MessageHeader header;
    header.set_message_type(message_type_);
    header.set_message_length(header_only_ ? 0 : msg->size());
    header.set_signature("MESSAGE");
    header.set_sender(self_name_);
    header.set_header_only(header_only_);
    header.set_send_time_ns(GetTimestampNs());
    if (!(SendProto(socket_fd_, header) &&
          (header.header_only() ||
           SendAll(socket_fd_, &(*msg)[0], msg->size())))) {
      socket_status_ = MessageSenderStatus::BROKEN;
      continue;
    }
    msgs_sent_++;
    bytes_sent_ += header.ByteSizeLong() + (header_only_ ? 0 : msg->size());
  }
}

void MessageSender::Connect() {
  while (!shutdown_ && socket_status_ != MessageSenderStatus::CONNECTED) {
    if (socket_status_ != MessageSenderStatus::NEW && socket_fd_ >= 0) {
      shutdown(socket_fd_, SHUT_RDWR);
      close(socket_fd_);
      disconnects_++;
    }

    socket_fd_ = socket(remote_addr_.GetAddressFamily(), SOCK_STREAM, 0);
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(socket_fd_, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<void *>(&timeout), sizeof(timeout));
    if (remote_addr_.GetAddressFamily() == AF_INET) {
      int nodelay = 1;
      setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<void *>(&nodelay), sizeof(nodelay));
    }
    socket_status_ = MessageSenderStatus::CONNECTING;

    if (connect(socket_fd_, remote_addr_.GetRawSockAddr(),
                remote_addr_.GetRawSockAddrLength()) < 0) {
      socket_status_ = MessageSenderStatus::CONNECTION_FAILED;
      if (!shutdown_) {
        sleep(1);
      }
    } else {
      queue_mutex_.lock();
      buffer_queue_.clear();
      queue_mutex_.unlock();
      socket_status_ = MessageSenderStatus::CONNECTED;
    }
  }
}

void MessageSender::Diagnose(MessageSenderStatus *status) {
  status->set_message_type(message_type_);
  status->set_remote_endpoint(endpoint_);
  status->set_status(socket_status_);
  status->set_msgs_enqueued(msgs_enqueued_);
  status->set_msgs_sent(msgs_sent_);
  status->set_bytes_sent(bytes_sent_);
  status->set_disconnects(disconnects_);

  queue_mutex_.lock();
  status->set_queue_size(buffer_queue_.size());
  queue_mutex_.unlock();
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
