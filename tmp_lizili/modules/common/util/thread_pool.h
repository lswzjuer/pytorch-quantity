#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(size_t thread_num, size_t task_num = 0);
  size_t UnfinishedTaskNum();
  template <class F, class... Args>
  auto Enqueue(F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers_;
  // the task queue
  std::queue<std::function<void()>> tasks_;

  // synchronization
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  size_t thread_num_;
  size_t task_num_;
  bool stop_;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t thread_num, size_t task_num)
    : thread_num_(thread_num), task_num_(task_num), stop_(false) {
  for (size_t i = 0; i < thread_num_; ++i) {
    workers_.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(
              lock, [this] { return this->stop_ || !this->tasks_.empty(); });
          if (this->stop_ && this->tasks_.empty()) return;
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }

        task();
      }
    });
  }
}

inline size_t ThreadPool::UnfinishedTaskNum() {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return tasks_.size();
}

/**
 * @brief Add new work item to the pool.
 * @attention If you set task num and task is pop before it is called, when you
 * use future.get(), it will throw std::future_error : Broken promise. You
 * should be handle it.
 */
template <class F, class... Args>
auto ThreadPool::Enqueue(F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // don't allow enqueueing after stopping the pool
    if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks_.emplace([task]() { (*task)(); });
    if (task_num_ > 0 && tasks_.size() > task_num_) {
      tasks_.pop();
    }
  }
  condition_.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (std::thread &worker : workers_) worker.join();
}

#endif
