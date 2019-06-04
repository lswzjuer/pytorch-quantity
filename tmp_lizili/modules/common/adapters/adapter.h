/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/**
 * @file
 */

#ifndef MODULES_ADAPTERS_ADAPTER_H_
#define MODULES_ADAPTERS_ADAPTER_H_

#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"

#include "modules/common/adapters/adapter_gflags.h"
#include "modules/common/adapters/adapter_utils.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/log.h"
#include "modules/common/proto/header.pb.h"
#include "modules/common/util/file.h"
#include "modules/common/util/string_util.h"
#include "modules/common/util/util.h"

/**
 * @namespace roadstar::common::adapter
 * @brief roadstar::common::adapter
 */
namespace roadstar {
namespace common {
namespace adapter {

/**
 * @class AdapterBase
 * @brief Base interface of all concrete adapters.
 */
class AdapterBase {
 public:
  virtual ~AdapterBase() = default;

  /**
   * @brief returns the topic name that this adapter listens to.
   */
  virtual const std::string &topic_name() const = 0;

  /**
   * @brief Create a view of data up to the call time for the user.
   */
  virtual void Observe() = 0;

  /**
   * @brief returns TRUE if the observing queue is empty.
   */
  virtual bool Empty() const = 0;

  /**
   * @brief returns TRUE if the adapter has received any message.
   */
  virtual bool HasReceived() const = 0;

  /**
   * @brief Gets message delay.
   */
  virtual double GetDelaySec() const = 0;

  /**
   * @brief Clear the data received so far.
   */
  virtual void ClearData() = 0;

  virtual void FeedBuffer(const std::vector<unsigned char> &buffer) = 0;

  virtual void TriggerHeaderOnlyCallbacks() = 0;
};

/**
 * @class Adapter
 * @brief this class serves as the interface and a layer of
 * abstraction for Roadstar modules to interact with various I/O (e.g.
 * ROS). The adapter will also store history data, so that other
 * Roadstar modules can get access to both the current and the past data
 * without having to handle communication protocols directly.
 *
 * \par
 * Each \class Adapter instance only works with one single topic and
 * its corresponding data type.
 *
 * \par
 * Under the hood, a queue is used to store the current and historical
 * messages. In most cases, the underlying data type is a proto, though
 * this is not necessary.
 *
 * \note
 * Adapter::Observe() is thread-safe, but calling it from
 * multiple threads may introduce unexpected behavior. Adapter is
 * thread-safe w.r.t. data access and update.
 */
template <typename D>
class Adapter : public AdapterBase {
 public:
  /// The user can use Adapter::DataType to get the type of the
  /// underlying data.
  typedef D DataType;

  typedef typename std::list<std::shared_ptr<D>>::const_iterator Iterator;
  typedef typename std::function<void(const D &)> Callback;
  typedef typename std::function<void()> HeaderOnlyCallback;

  /**
   * @brief Construct the \class Adapter object.
   * @param adapter_name the name of the adapter. It is used to log
   * error messages when something bad happens, to help people get an
   * idea which adapter goes wrong.
   * @param topic_name the topic that the adapter listens to.
   * @param message_num the number of historical messages that the
   * adapter stores. Older messages will be removed upon calls to
   * Adapter::OnReceive().
   */
  Adapter(const std::string &adapter_name, const std::string &topic_name,
          size_t message_num, const std::string &dump_dir = "/tmp")
      : topic_name_(topic_name), message_num_(message_num) {}

  /**
   * @brief returns the topic name that this adapter listens to.
   */
  const std::string &topic_name() const override {
    return topic_name_;
  }

  /**
   * @brief the callback that will be invoked whenever a new
   * message is received.
   * @param message the newly received message.
   */
  void OnReceive(const D &message) {
    last_receive_time_ = roadstar::common::time::Clock::NowInSecond();
    EnqueueData(message);
    FireCallbacks(message);
  }

  /**
   * @brief copy the data_queue_ into the observing queue to create a
   * view of data up to the call time for the user.
   */
  void Observe() override {
    std::lock_guard<std::mutex> lock(mutex_);
    observed_queue_ = data_queue_;
  }

  /**
   * @brief returns TRUE if the observing queue is empty.
   */
  bool Empty() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return observed_queue_.empty();
  }

  /**
   * @brief returns TRUE if the adapter has received any message.
   */
  bool HasReceived() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return !data_queue_.empty();
  }

  /**
   * @brief returns the most recent message in the observing queue.
   *
   * /note
   * Please call Empty() to make sure that there is data in the
   * queue before calling GetOldestObserved().
   */
  const D &GetLatestObserved() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DCHECK(!observed_queue_.empty())
        << "The view of data queue is empty. No data is received yet or you "
           "forgot to call Observe()"
        << ":" << topic_name_;
    return *observed_queue_.front();
  }

  /**
   * @brief returns the most recent message given a timestamp in sec.
   *
   * /note
   * consider may be call in multi-thread, we get realtime data queue
   * may be timestamp order is not uncertain in data queue. so we travel
   * all observed queue.
   */
  const D &GetExpectedObserved(
      double time_sec,
      std::function<bool(const D &)> check = [](const D &obj) -> bool {
        return true;
      }) {
    auto ptr = GetExpectedObservedPtr(time_sec, check);
    static D null_data;
    if (!ptr) {
      return null_data;
    } else {
      return *ptr;
    }
  }

  std::shared_ptr<const D> GetExpectedObservedPtr(
      const QueryTime &query_time,
      std::function<bool(const D &)> check = [](const D &obj) -> bool {
        return true;
      }) {
    double time_start = common::time::Clock::NowInSecond();
    while (true) {
      const auto value = GetExpectedObservedPtr(query_time.time_sec, check);
      if (value &&
          fabs(GetMsgTimeInSec<D>(*value) - query_time.time_sec) < 1e-6) {
        return value;
      } else {
        double time_now = common::time::Clock::NowInSecond();
        if (time_now - time_start > query_time.max_wait_time) {
          return nullptr;
        }
        sleep(static_cast<unsigned int>(query_time.query_step_time));
      }
    }
  }

  std::shared_ptr<const D> GetExpectedObservedPtr(
      double time_sec,
      std::function<bool(const D &)> check = [](const D &obj) -> bool {
        return true;
      }) {
    mutex_.lock();
    std::list<std::shared_ptr<D>> observed_queue(data_queue_.begin(),
                                                 data_queue_.end());
    mutex_.unlock();
    if (observed_queue.empty()) {
      return nullptr;
    }
    double min_error = std::numeric_limits<double>::max();
    Iterator expected_it = observed_queue.begin();
    for (Iterator it = observed_queue.begin(); it != observed_queue.end();
         it++) {
      double check_time = GetMsgTimeInSec<D>(*(*it));
      double now_error = fabs(check_time - time_sec);
      if (now_error < min_error && check(**it)) {
        min_error = now_error;
        expected_it = it;
      }
    }
    if (check(*(*expected_it))) {
      return *expected_it;
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Gets message delay.
   */
  double GetDelaySec() const override {
    if (last_receive_time_ == 0) {
      return -1;
    } else {
      return roadstar::common::time::Clock::NowInSecond() - last_receive_time_;
    }
  }

  /**
   * @brief Clear the data received so far.
   */
  void ClearData() override {
    // Lock the queue.
    std::lock_guard<std::mutex> lock(mutex_);
    data_queue_.clear();
    observed_queue_.clear();
  }

  /**
   * @brief returns the oldest message in the observing queue.
   *
   * /note
   * Please call Empty() to make sure that there is data in the
   * queue before calling GetOldestObserved().
   */
  const D &GetOldestObserved() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DCHECK(!observed_queue_.empty())
        << "The view of data queue is empty. No data is received yet or you "
           "forgot to call Observe().";
    return *observed_queue_.back();
  }

  /**
   * @brief returns an iterator representing the head of the observing
   * queue. The caller can use it to iterate over the observed data
   * from the head. The API also supports range based for loop.
   */
  Iterator begin() const {
    return observed_queue_.begin();
  }

  /**
   * @brief returns an iterator representing the tail of the observing
   * queue. The caller can use it to iterate over the observed data
   * from the head. The API also supports range based for loop.
   */
  Iterator end() const {
    return observed_queue_.end();
  }

  /**
   * @brief registers the provided callback function to the adapter,
   * so that the callback function will be called once right after the
   * message hits the adapter.
   * @param callback the callback with signature void(const D &).
   */
  void AddCallback(Callback callback) {
    receive_callbacks_.push_back(callback);
  }

  void AddHeaderOnlyCallback(HeaderOnlyCallback callback) {
    receive_header_callbacks_.push_back(callback);
  }

  /**
   * @brief fills the fields module_name, timestamp_sec and
   * sequence_num in the header.
   */
  void FillHeader(const std::string &module_name, D *data) {
    double timestamp = roadstar::common::time::Clock::NowInSecond();
    FillHeader(module_name, timestamp, data);
  }

  /**
   * @brief assign timestamp_sec, fills the fields module_name and
   * sequence_num in the header.
   */
  void FillHeader(const std::string &module_name, const double sensor_time,
                  D *data) {
    static_assert(std::is_base_of<google::protobuf::Message, D>::value,
                  "Can only fill header to proto messages!");
    auto *header = data->mutable_header();
    double timestamp = roadstar::common::time::Clock::NowInSecond();
    header->set_module_name(module_name);
    header->set_timestamp_sec(sensor_time);
    header->set_pub_timestamp_sec(timestamp);
    header->set_sequence_num(++seq_num_);
  }

  uint32_t GetSeqNum() const {
    return seq_num_;
  }

  void FeedBuffer(const std::vector<unsigned char> &buffer) override {
    D data;
    Deserialize<D>(buffer, &data);
    OnReceive(data);
  }

  void TriggerHeaderOnlyCallbacks() override {
    for (const auto &callback : receive_header_callbacks_) {
      callback();
    }
  }

 private:
  /**
   * @brief proactively invokes the callbacks one by one registered with the
   * specified data.
   * @param data the specified data.
   */
  void FireCallbacks(const D &data) {
    for (const auto &callback : receive_callbacks_) {
      callback(data);
    }
  }

  /**
   * @brief push the shared-pointer-guarded data to the data queue of
   * the adapter.
   */
  void EnqueueData(const D &data) {
    // Don't try to copy data and enqueue if the message_num is 0
    if (message_num_ == 0) {
      return;
    }

    // Lock the queue.
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_queue_.size() + 1 > message_num_) {
      data_queue_.pop_back();
    }
    data_queue_.push_front(std::make_shared<D>(data));
  }

  /// The topic name that the adapter listens to.
  std::string topic_name_;

  /// The maximum size of data_queue_ and observed_queue_
  size_t message_num_ = 0;

  /// The received data. Its size is no more than message_num_
  std::list<std::shared_ptr<D>> data_queue_;

  /// It is the snapshot of the data queue. The snapshot is taken when
  /// Observe() is called.
  std::list<std::shared_ptr<D>> observed_queue_;

  /// User defined function when receiving a message
  std::vector<Callback> receive_callbacks_;

  /// User defined function when receiving a message with only its header
  std::vector<HeaderOnlyCallback> receive_header_callbacks_;

  /// The mutex guarding data_queue_ and observed_queue_
  mutable std::mutex mutex_;

  /// The monotonically increasing sequence number of the message to
  /// be published.
  uint32_t seq_num_ = 0;

  double last_receive_time_ = 0;
};

}  // namespace adapter
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_ADAPTERS_ADAPTER_H_
