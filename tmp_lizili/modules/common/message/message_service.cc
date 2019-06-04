#include "modules/common/message/message_service.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

#include "modules/common/log.h"
#include "modules/common/message/message_receiver.h"
#include "modules/common/message/utils.h"
#include "modules/common/module_conf/module_util.h"
#include "modules/common/util/file.h"
#include "modules/msgs/module_conf/proto/module_conf.pb.h"

namespace roadstar {
namespace common {
namespace message {
namespace {
std::once_flag once_flag;
static const int kMessageServiceWakeSignal = SIGRTMIN + 3;
void SigHandler(int sig) {
  // Do nothing
}
void RegisterSigHandler() {
  AFATAL_IF(signal(kMessageServiceWakeSignal, &SigHandler) != nullptr)
      << "Conflicting real-time signal: " << kMessageServiceWakeSignal;
}
}  // namespace

MessageService::MessageService() {}

MessageService::~MessageService() {
  ShutDown();
}

void MessageService::ShutDown() {
  bool expect = false;
  if (shutdown_.compare_exchange_strong(expect, true)) {
    shutdown(accept_fd_, SHUT_RDWR);
    close(accept_fd_);

    if (accept_thread_) {
      accept_thread_->join();
      accept_thread_.reset(nullptr);
    }
    if (diagnose_thread_) {
      pthread_kill(diagnose_thread_->native_handle(),
                   kMessageServiceWakeSignal);
      diagnose_thread_->join();
      diagnose_thread_.reset(nullptr);
    }

    std::lock_guard<std::mutex> lock(receivers_mutex_);
    topic_endpoint_map_.clear();
    publishing_types_.clear();
    message_receivers_.clear();
  }
}

void MessageService::Init(
    const std::string &module_name,
    std::function<void(adapter::AdapterConfig::MessageType,
                       const std::vector<unsigned char> &buffer, bool)>
        callback) {
  instance()->InitImpl(module_name, callback);
}

void MessageService::InitImpl(
    const std::string &module_name,
    std::function<void(adapter::AdapterConfig::MessageType,
                       const std::vector<unsigned char> &buffer, bool)>
        callback) {
  if (initialized_) {
    return;
  }
  std::string lockfile = "/tmp/" + module_name + ".lockfile";
  int fd = open(lockfile.c_str(), O_RDWR | O_CREAT, 0666);
  flock file_lock;
  file_lock.l_type = F_WRLCK;
  file_lock.l_whence = SEEK_SET;
  file_lock.l_start = 0;
  file_lock.l_len = 0;
  if (fcntl(fd, F_SETLK, &file_lock) == -1) {
    AFATAL << strerror(errno) << ": maybe another module with the same name "
           << module_name << " is already running.";
  }
  shutdown_ = false;
  module_name_ = module_name;
  callback_ = callback;
  endpoint_ = module_conf::ModuleUtil::GetEndpoint(module_name);
  publishing_types_ = module_conf::ModuleUtil::GetPublishingTypes(module_name);
  publishing_types_.insert(adapter::AdapterConfig::MESSAGE_SERVICE_STATUS);

  for (auto endpoint :
       module_conf::ModuleUtil::GetRemoteEndpoints(publishing_types_)) {
    topic_endpoint_map_.insert(std::make_pair(
        endpoint.message_type,
        std::make_pair(
            endpoint.module_name,
            std::unique_ptr<MessageSender>(new MessageSender(
                static_cast<adapter::AdapterConfig::MessageType>(
                    endpoint.message_type),
                endpoint.endpoint, module_name, endpoint.header_only)))));
    AINFO << adapter::AdapterConfig::MessageType_descriptor()
                 ->FindValueByNumber(endpoint.message_type)
                 ->name()
          << " -> " << endpoint.module_name << ":" << endpoint.endpoint;
  }
  std::call_once(once_flag, RegisterSigHandler);
  accept_thread_.reset(
      new std::thread(std::bind(&MessageService::AcceptThread, this)));
  diagnose_thread_.reset(
      new std::thread(std::bind(&MessageService::DiagnoseThread, this)));
  initialized_ = true;
}

void MessageService::SendImpl(
    adapter::AdapterConfig::MessageType message_type,
    std::shared_ptr<std::vector<unsigned char>> buffer) {
  if (!publishing_types_.count(message_type)) {
    AFATAL << "Sending unconfigured message type: "
           << adapter::AdapterConfig::MessageType_descriptor()
                  ->FindValueByNumber(message_type)
                  ->name();
  }
  auto range = topic_endpoint_map_.equal_range(message_type);
  for (auto it = range.first; it != range.second; ++it) {
    it->second.second->Send(buffer);
  }
}

void MessageService::AcceptThread() {
  if (endpoint_.empty()) {
    return;
  }
  SockAddr addr(endpoint_);
  accept_fd_ = socket(addr.GetAddressFamily(), SOCK_STREAM, 0);

  int so_reuseaddr = 1;
  setsockopt(accept_fd_, SOL_SOCKET, SO_REUSEADDR, &so_reuseaddr,
             sizeof(so_reuseaddr));

  if (Bind(accept_fd_, addr.GetRawSockAddr(), addr.GetRawSockAddrLength()) <
      0) {
    AFATAL << endpoint_ << " bind failed: " << strerror(errno);
  }

  if (listen(accept_fd_, 128) < 0) {
    AFATAL << endpoint_ << " listen failed: " << strerror(errno);
  }

  while (!shutdown_) {
    SockAddr peer(addr.GetAddressFamily());
    socklen_t peer_len = addr.GetRawSockAddrLength();
    int rfd = ::accept(accept_fd_, peer.GetRawSockAddr(), &peer_len);
    if (rfd < 0) {
      AWARN << endpoint_ << " accept failed: " << strerror(errno);
      continue;
    }
    ADEBUG << "Incoming fd: " << rfd;
    auto *receiver = new MessageReceiver(rfd, callback_);
    receivers_mutex_.lock();
    message_receivers_.push_back(std::unique_ptr<MessageReceiver>(receiver));
    receivers_mutex_.unlock();
    // TODO(wanxinyi): Cleanup broken receivers;
  }
}

void MessageService::DiagnoseThread() {
  MessageServiceStatus status;
  while (!shutdown_) {
    Diagnose(&status);
    Send(adapter::AdapterConfig::MESSAGE_SERVICE_STATUS, status);
    sleep(1);
  }
}

void MessageService::Diagnose(MessageServiceStatus *status) {
  status->Clear();
  timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  status->set_timestamp_ns(now.tv_sec * 1000000000ll + now.tv_nsec);

  status->set_endpoint(endpoint_);
  status->set_module_name(module_name_);
  for (const auto &it : topic_endpoint_map_) {
    auto *sender = status->add_senders();
    sender->set_target_module(it.second.first);
    it.second.second->Diagnose(sender);
  }
  receivers_mutex_.lock();
  for (const auto &receiver : message_receivers_) {
    if (receiver->IsConnected()) {
      receiver->Diagnose(status->add_receivers());
    }
  }
  receivers_mutex_.unlock();
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
