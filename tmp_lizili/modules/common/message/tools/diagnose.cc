#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <time.h>
#include <csignal>
#include <functional>
#include <iomanip>

#include "gflags/gflags.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/message/message_service.h"
#include "modules/common/message/proto/diagnose.pb.h"
#include "modules/common/message/proto/message_header.pb.h"
#include "modules/common/message/utils.h"
#include "modules/common/module_conf/module_util.h"
#include "modules/common/util/file.h"
#include "modules/msgs/module_conf/proto/module_conf.pb.h"

using roadstar::common::adapter::AdapterConfig;
using roadstar::common::message::MessageHeader;
using roadstar::common::message::MessageReceiverStatus;
using roadstar::common::message::MessageSenderStatus;
using roadstar::common::message::MessageService;
using roadstar::common::message::MessageServiceStatus;

DEFINE_string(module_name, "", "module_name");
DEFINE_bool(bag_only, false, "only show status form recorded bag");
DEFINE_bool(external, false, "only show external living modules");

namespace {

const char kRed[] = "\e[41m";
const char kReset[] = "\e[0m";
const std::unordered_map<std::string, std::string> kColorMap = {
    {"CONNECTION_FAILED", kRed}, {"CONNECTING", kRed}, {"BROKEN", kRed}};

void DisplayRows(std::vector<std::vector<std::string>> rows) {
  if (rows.empty()) {
    return;
  }
  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  int screen_width = w.ws_col;
  std::vector<int> col_width;
  for (size_t i = 0; i < rows[0].size(); i++) {
    col_width.push_back(0);
    for (size_t j = 0; j < rows.size(); j++) {
      size_t len = rows[j][i].length() + 2;
      col_width[i] = std::max<int>(col_width[i], len);
    }
  }
  size_t col_start = 0;
  while (col_start < rows[0].size()) {
    size_t col_end = col_start;
    int sum = 0;
    while (true) {
      if (col_end >= rows[0].size() ||
          sum + col_width[col_end] > screen_width) {
        break;
      }
      sum += col_width[col_end];
      col_end++;
    }
    for (auto row : rows) {
      for (size_t j = col_start; j < col_end; j++) {
        bool colored = kColorMap.count(row[j]);
        if (colored) {
          std::cout << kColorMap.at(row[j]);
        }
        std::cout << std::internal << std::setw(col_width[j]) << row[j];
        if (colored) {
          std::cout << kReset;
        }
      }
      std::cout << std::endl;
    }
    col_start = col_end;
  }
}
void DisplaySenders(const MessageServiceStatus &status) {
  std::vector<
      std::pair<std::string, std::function<std::string(MessageSenderStatus)>>>
      sender_funcs = {
          {"MessageType",
           [](const MessageSenderStatus status) {
             return roadstar::common::adapter::AdapterConfig::
                 MessageType_descriptor()
                     ->FindValueByNumber(status.message_type())
                     ->name();
           }},
          {"Target", &MessageSenderStatus::target_module},
          {"Status",
           [](const MessageSenderStatus status) {
             std::string res = MessageSenderStatus::Status_descriptor()
                                   ->FindValueByNumber(status.status())
                                   ->name();
             return res;
           }},
          {"Target Endpoint", &MessageSenderStatus::remote_endpoint},
          {"Msgs Sent",
           [](const MessageSenderStatus status) {
             return std::to_string(status.msgs_sent());
           }},
          {"Msgs Enqueued",
           [](const MessageSenderStatus status) {
             return std::to_string(status.msgs_enqueued());
           }},
      };

  std::vector<std::vector<std::string>> rows;
  for (const auto &funcs : sender_funcs) {
    rows.push_back(std::vector<std::string>());
    rows.back().push_back(funcs.first);
    for (const auto &sender : status.senders()) {
      if (sender.message_type() == AdapterConfig::MESSAGE_SERVICE_STATUS) {
        continue;
      }
      rows.back().push_back(funcs.second(sender));
    }
  }
  DisplayRows(rows);
}

void DisplayReceivers(const MessageServiceStatus &status) {
  std::vector<std::function<std::string(MessageReceiverStatus)>>
      receiver_funcs = {
          [](const MessageReceiverStatus status) {
            return roadstar::common::adapter::AdapterConfig::
                MessageType_descriptor()
                    ->FindValueByNumber(status.message_type())
                    ->name();
          },
          &MessageReceiverStatus::remote_name,
          [](const MessageReceiverStatus status) {
            return std::to_string(status.msgs_received());
          },
      };

  std::vector<std::vector<std::string>> rows;
  for (const auto &funcs : receiver_funcs) {
    rows.push_back(std::vector<std::string>());
    for (const auto &receiver : status.receivers()) {
      if (receiver.has_received()) {
        rows.back().push_back(funcs(receiver));
      }
    }
  }
  DisplayRows(rows);
}

void ClearInternalModules(MessageServiceStatus *status) {
  auto modules =
      roadstar::common::module_conf::ModuleUtil::GetExternalModuleNames();
  for (auto it = status->mutable_senders()->begin();
       it != status->mutable_senders()->end();) {
    if (!modules.count(it->target_module())) {
      status->mutable_senders()->erase(it);
    } else {
      it++;
    }
  }
  for (auto it = status->mutable_receivers()->begin();
       it != status->mutable_receivers()->end();) {
    if (!modules.count(it->remote_name())) {
      status->mutable_receivers()->erase(it);
    } else {
      it++;
    }
  }
}

void DisplayStatus(const MessageServiceStatus &status) {
  if (status.module_name() != FLAGS_module_name) {
    return;
  }
  if (FLAGS_external) {
    ClearInternalModules(const_cast<MessageServiceStatus *>(&status));
  }
  std::cout << status.timestamp_ns() << std::endl;
  std::cout << status.module_name() << std::endl;
  std::cout << status.endpoint() << std::endl;
  std::cout << "------senders" << std::endl;
  DisplaySenders(status);
  std::cout << "------receivers" << std::endl;
  DisplayReceivers(status);
}

void Callback(AdapterConfig::MessageType type,
              const std::vector<unsigned char> &buffer, bool header_only) {
  MessageServiceStatus status;
  status.ParseFromArray(&buffer[0], buffer.size());
  timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  if (now.tv_sec - status.timestamp_ns() / 1000000000ll < 10) {
    return;
  }
  DisplayStatus(status);
}

}  // namespace

int main(int argc, char *argv[]) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_bag_only) {
    MessageService::Init("diagnose", Callback);
    while (true) {
      sleep(1);
    }
  } else {
    std::string endpoint =
        roadstar::common::module_conf::ModuleUtil::GetEndpoint(
            FLAGS_module_name);

    if (endpoint.empty()) {
      AFATAL << "Invalid module: " << FLAGS_module_name;
    }
    int socket_fd;
    roadstar::common::message::SockAddr remote(endpoint);
    socket_fd = socket(remote.GetAddressFamily(), SOCK_STREAM, 0);
    if (connect(socket_fd, remote.GetRawSockAddr(),
                remote.GetRawSockAddrLength()) < 0) {
      AFATAL << "Connection Failded to " << endpoint << ": " << strerror(errno);
    }
    MessageHeader header;
    header.set_signature("MESSAGE");
    header.set_diagnose(true);
    SendProto(socket_fd, header);

    MessageServiceStatus status;
    while (true) {
      RecvProto(socket_fd, &status) || AFATAL << "Connection broken";
      DisplayStatus(status);
    }
  }
}
