/******************************************************************************
 * Base realization
 *****************************************************************************/
#ifndef MODULES_COMMON_UTIL_CONCURRENT_QUEUE_H_
#define MODULES_COMMON_UTIL_CONCURRENT_QUEUE_H_

#include <condition_variable>
#include <deque>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <utility>

namespace roadstar {
namespace common {

template <typename T>
class ConcurrentQueue final {
 public:
  ConcurrentQueue() = default;
  explicit ConcurrentQueue(size_t capacity) : capacity_(capacity) {}

  ~ConcurrentQueue() {}

  template <typename U = T>
  std::enable_if_t<std::is_convertible_v<U, T>, void> Push(U &&new_data) {
    std::lock_guard lk(mutex_);
    data_queue_.emplace(std::forward<U>(new_data));
    if (capacity_ > 0 && data_queue_.size() > capacity_) {
      data_queue_.pop();
    }
    cond_.notify_one();
  }

  template <typename... Args>
  void Push(std::in_place_t, Args &&... args) {
    std::lock_guard lk(mutex_);
    data_queue_.emplace(std::forward<Args>(args)...);
    if (capacity_ > 0 && data_queue_.size() > capacity_) {
      data_queue_.pop();
    }
    cond_.notify_one();
  }

  template <typename... Args>
  void Emplace(Args &&... args) {
    Push(std::in_place, std::forward<Args>(args)...);
  }

  void Pop(T *val) {
    std::unique_lock ulk(mutex_);
    cond_.wait(ulk, [this]() { return !data_queue_.empty(); });
    *val = std::move(data_queue_.front());
    data_queue_.pop();
  }

  T Pop() {
    std::unique_lock ulk(mutex_);
    cond_.wait(ulk, [this]() { return !data_queue_.empty(); });
    auto tmp = std::move(data_queue_.front());
    data_queue_.pop();
    return std::move(tmp);
  }

  bool TryFront(T *val) const {
    std::shared_lock slk(mutex_);
    if (data_queue_.empty()) {
      return false;
    }
    *val = data_queue_.front();
    return true;
  }

  T Front() const {
    std::shared_lock slk(mutex_);
    cond_.wait(slk, [this]() { return !data_queue_.empty(); });
    return data_queue_.front();
  }

  bool TryPop(T *val) {
    std::lock_guard lk(mutex_);
    if (data_queue_.empty()) {
      return false;
    }
    *val = std::move(data_queue_.front());
    data_queue_.pop();
    return true;
  }

  inline bool Empty() const {
    std::lock_guard lk(mutex_);
    return data_queue_.empty();
  }

  inline size_t Size() const {
    std::lock_guard lk(mutex_);
    return data_queue_.size();
  }

  inline size_t Capacity() const {
    std::lock_guard lk(mutex_);
    return capacity_;
  }

  void Reserve(size_t capacity) const {
    capacity_ = capacity;
  }

 private:
  std::queue<T> data_queue_;
  mutable std::shared_mutex mutex_;
  std::condition_variable_any cond_;
  size_t capacity_ = 0;
};

}  // namespace common
}  // namespace roadstar
#endif  // MODULES_COMMON_UTIL_CONCURRENT_QUEUE_H_
