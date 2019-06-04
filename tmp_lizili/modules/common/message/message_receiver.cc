#include "modules/common/message/message_receiver.h"

#include <arpa/inet.h>
#include <endian.h>
#include <netinet/tcp.h>
#include <semaphore.h>
#include <cstring>

#include "modules/common/log.h"
#include "modules/common/message/message_service.h"
#include "modules/common/message/proto/message_header.pb.h"
#include "modules/common/message/utils.h"

namespace roadstar {
namespace common {
namespace message {
namespace {
std::once_flag once_flag;
static const int kMessageReceiverWakeSignal = SIGRTMIN + 2;
void SigHandler(int sig) {
  // Do nothing
}
void RegisterSigHandler() {
  AFATAL_IF(signal(kMessageReceiverWakeSignal, &SigHandler) != nullptr)
      << "Conflicting real-time signal: " << kMessageReceiverWakeSignal;
}
}  // namespace

MessageReceiver::MessageReceiver(int fd, const Callback &callback)
    : fd_(fd),
      callback_(callback),
      connected_(true),
      msgs_received_(0),
      bytes_received_(0) {
  std::call_once(once_flag, RegisterSigHandler);
  thread_.reset(new std::thread(std::bind(&MessageReceiver::RunLoop, this)));
}

MessageReceiver::~MessageReceiver() {
  connected_ = false;
  shutdown(fd_, SHUT_RDWR);
  close(fd_);
  // Wake threads
  pthread_kill(thread_->native_handle(), kMessageReceiverWakeSignal);
  thread_->join();
}

void MessageReceiver::RunLoop() {
  while (connected_) {
    // Parse header
    MessageHeader message_header;
    if (!RecvProto(fd_, &message_header) ||
        message_header.signature() != "MESSAGE") {
      ADEBUG << "Corrupted data received";
      connected_ = false;
      shutdown(fd_, SHUT_RDWR);
      break;
    }

    if (message_header.diagnose()) {
      while (connected_) {
        MessageServiceStatus status;
        MessageService::instance()->Diagnose(&status);
        if (!SendProto(fd_, status)) {
          break;
        }
        sleep(1);
      }
      connected_ = false;
      shutdown(fd_, SHUT_RDWR);
      break;
    }

    std::vector<unsigned char> buffer(message_header.message_length());
    if (!message_header.header_only() &&
        !RecvAll(fd_, &buffer[0], message_header.message_length())) {
      ADEBUG << "Receiving data failed";
      connected_ = false;
      shutdown(fd_, SHUT_RDWR);
      break;
    }
    msgs_received_++;
    bytes_received_ += message_header.ByteSizeLong() + buffer.size();
    message_type_ = static_cast<adapter::AdapterConfig::MessageType>(
        message_header.message_type());
    remote_name_ = message_header.sender();
    // TODO(wanxinyi): invoke callback on worker threads with shared_ptr
    callback_(static_cast<adapter::AdapterConfig::MessageType>(
                  message_header.message_type()),
              buffer, message_header.header_only());
  }
}

void MessageReceiver::Diagnose(MessageReceiverStatus *status) {
  status->set_msgs_received(msgs_received_);
  status->set_bytes_received(bytes_received_);
  if (msgs_received_ > 0) {
    status->set_message_type(message_type_);
    status->set_has_received(true);
    status->set_remote_name(remote_name_);
  }
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
