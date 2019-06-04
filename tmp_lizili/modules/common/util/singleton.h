#pragma once
namespace roadstar::common::util {

template <typename T, typename CONTEXT>
class Singleton {
 public:
  T *operator->() {
    return instance_;
  }
  const T *operator->() const {
    return instance_;
  }
  T &operator*() {
    return *instance_;
  }
  const T &operator*() const {
    return *instance_;
  }

  static T *instance() {
    return instance_;
  }

 protected:
  Singleton() {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
    static bool static_init = []() -> bool {
      instance_ = new T;
      return true;
    }();
  }
#pragma clang diagnostic pop

 private:
  static T *instance_;
};

template <typename T, typename CONTEXT>
T *Singleton<T, CONTEXT>::instance_;

}  // namespace roadstar::common::util
