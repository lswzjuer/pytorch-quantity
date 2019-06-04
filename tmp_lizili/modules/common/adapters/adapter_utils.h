#ifndef MODULES_ADAPTERS_ADAPTER_UTILS_H_
#define MODULES_ADAPTERS_ADAPTER_UTILS_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ros/ros.h"
#include "rosbag/bag.h"

#include "modules/common/time/time.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace common {
namespace adapter {

#define REGISTER_SHARED_ADAPTER(name)                                 \
 public:                                                              \
  static Adapter<name>* Get##name() {                                 \
    return instance()->InternalGet##name();                           \
  }                                                                   \
                                                                      \
 private:                                                             \
  void Enable##name(const std::string& topic_name,                    \
                    int message_history_limit) {                      \
    name##_.reset(                                                    \
        new Adapter<name>(#name, topic_name, message_history_limit)); \
  }                                                                   \
  Adapter<name>* InternalGet##name() {                                \
    return name##_.get();                                             \
  }                                                                   \
  std::unique_ptr<Adapter<name>> name##_;

// Borrowed from C++ 14.
template <bool B, class T = void>
using enable_if_t[[deprecated]] = typename std::enable_if<B, T>::type;

class SharedData {
 public:
  double timestamp_sec() const {
    return timestamp_sec_;
  }
  void set_timestamp_sec(double timestamp_sec) {
    timestamp_sec_ = timestamp_sec;
  }
  virtual ~SharedData() {}

 protected:
  double timestamp_sec_ = 0;
};

struct QueryTime {
  QueryTime(double time_sec, double max_wait_time,
            double query_step_time = 0.005)
      : time_sec(time_sec),
        max_wait_time(max_wait_time),
        query_step_time(query_step_time) {}
  double time_sec;
  double max_wait_time;
  double query_step_time;
};

#define IS_PROTOBUF \
  std::is_base_of<google::protobuf::Message, InputMessageType>::value

#define IS_SHARED_DATA std::is_base_of<SharedData, InputMessageType>::value

#define IS_ROSMSG                                                         \
  !std::is_base_of<google::protobuf::Message, InputMessageType>::value && \
      !std::is_base_of<SharedData, InputMessageType>::value

template <typename InputMessageType>
double GetMsgTimeInSec(enable_if_t<IS_ROSMSG, const InputMessageType&> data) {
  return data.header.stamp.sec + data.header.stamp.nsec * 1e-9;
}

template <typename InputMessageType>
double GetMsgTimeInSec(
    enable_if_t<IS_SHARED_DATA, const InputMessageType&> data) {
  return data.timestamp_sec();
}

template <typename InputMessageType>
double GetMsgTimeInSec(enable_if_t<IS_PROTOBUF, const InputMessageType&> data) {
  return data.header().timestamp_sec();
}

template <typename InputMessageType>
void Deserialize(const std::vector<unsigned char>& buffer,
                 enable_if_t<IS_SHARED_DATA, InputMessageType>* data) {
  return;
}

template <typename InputMessageType>
void Deserialize(const std::vector<unsigned char>& buffer,
                 enable_if_t<IS_PROTOBUF, InputMessageType>* data) {
  static_assert(IS_PROTOBUF, "Can only fill header to proto messages!");
  data->ParseFromArray(&buffer[0], buffer.size());
}

template <typename InputMessageType>
void Deserialize(const std::vector<unsigned char>& buffer,
                 enable_if_t<IS_ROSMSG, InputMessageType>* data) {
  ros::serialization::IStream stream(const_cast<unsigned char*>(&buffer[0]),
                                     buffer.size());
  ros::serialization::deserialize(stream, *data);
}

template <typename InputMessageType>
void GetEstimatedMsg(const double timestamp_sec, InputMessageType* data) {}

template <>
void GetEstimatedMsg(const double timestamp_sec,
                     ::roadstar::localization::Localization* loc);

}  // namespace adapter
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_ADAPTERS_ADAPTER_UTILS_H_
