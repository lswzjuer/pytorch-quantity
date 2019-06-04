#include "modules/common/message/tools/bag_tool.h"

#include <endian.h>
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>

#include "modules/common/log.h"

namespace roadstar {
namespace common {
namespace message {

BagPlayer::BagPlayer(
    const std::vector<std::string>& filenames, 
    const std::string& ignore_list)
    : paused_(false), step_(false), onplay_(false) {
  std::unordered_set<int> ignore_set;
  ParseTypeList(ignore_list, &ignore_set);
  reader_.reset(new BagReader(filenames, ignore_set));
}

BagPlayer::~BagPlayer() {
  FinishPlay(true);
  play_thread_.reset();
}

void BagPlayer::Launch(double start_time, double duration_time,
    double speed_rate, bool loop, OnChunkReady callback) {
  if (onplay_) {
    AFATAL << "Already launched!";
  }
  if (speed_rate <= 0.001) {
    speed_rate = 1;
  }
  if (start_time < 0) {
    start_time = 0;
  }
  uint64_t bag_length = reader_->GetEndTimeNs() - reader_->GetStartTimeNs();
  bag_start_time_ = start_time * 1000000000ll;
  bag_end_time_ = (start_time + duration_time) * 1000000000ll;
  if (duration_time < 0 || bag_end_time_ > bag_length) {
    bag_end_time_ = bag_length;
  }
  if (bag_start_time_ > bag_length) {
    AFATAL << "Start time bigger than bag length";
  }
  speed_rate_ = speed_rate;
  callback_ = callback;

  onplay_ = true;
  play_thread_.reset(new std::thread(
      std::bind(&BagPlayer::PlayThread, this, loop)));
}

void BagPlayer::ShowBagInfo() {
  std::map<int, int> type_count = reader_->GetTypeCount();
  for (auto it = type_count.begin(); it != type_count.end(); it++) {
    std::cout << adapter::AdapterConfig::MessageType_descriptor()
                 ->FindValueByNumber(it->first)
                 ->name()
          << ":" << it->second << std::endl;
  }
}

void BagPlayer::TogglePaused() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!paused_) {
    paused_ = true;
  } else {
    paused_ = false;
    lock.unlock();
    cond_.notify_one();
  }
}

void BagPlayer::StepAhead() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (paused_) {
    step_ = true;
    lock.unlock();
    cond_.notify_one();
  }
}

void BagPlayer::FinishPlay(bool force) {
  if (force) {
    onplay_ = false;
    StepAhead();
  }
  if (play_thread_->joinable()) {
    play_thread_->join();
  }
}

uint64_t BagPlayer::WaitAndContinue() {
  static uint64_t start_wait = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  if (!paused_ || step_) {
    step_ = false;
    return 0;
  }
  start_wait = GetTimestampNs();
  cond_.wait(lock, [this] () {return !paused_ || step_;}); 
  step_ = false;
  return GetTimestampNs() - start_wait;
}

void BagPlayer::PlayThread(bool loop) {
  BagDataChunk chunk;
  while (onplay_) {
    reader_->ResetToTimestamp(bag_start_time_);
    uint64_t real_start_time = GetTimestampNs();
    while (onplay_ && reader_->Next(&chunk)) {
      uint64_t delta = chunk.data_header().receive_time_ns() -
                       reader_->GetStartTimeNs();
      if (delta > bag_end_time_) {
        break;
      }
      delta -= bag_start_time_;
      delta /= speed_rate_;
      uint64_t eta = delta + real_start_time;
      SleepUntil(eta);
      real_start_time += WaitAndContinue();
      callback_(chunk, 
                reader_->GetStartTimeNs() + bag_start_time_,
                reader_->GetStartTimeNs() + bag_end_time_);
    }
    onplay_ = loop & onplay_;
  }
}

BagRecorder::BagRecorder(const size_t queue_size)
    : queue_size_(queue_size), writable_(true){
  write_thread_.reset(
    new std::thread(std::bind(&BagRecorder::WriteThread, this)));
}

void BagRecorder::Open(const std::string& filename) {
  std::unique_lock<std::mutex> lock(writer_mutex_);
  std::unique_ptr<BagWriter> finshed_writer= std::move(writer_);
  writer_.reset(new BagWriter(filename));
  lock.unlock();
  if(finshed_writer) {
    finshed_writer->Close();
  }
}

void BagRecorder::Close() {
  if (write_thread_){
    writable_ = false;
    cond_.notify_one();
    write_thread_->join();
    write_thread_.reset();
  }
  if (writer_) {
    writer_->Close();
    writer_.reset();
  }
}

void BagRecorder::FeedData(const BagDataChunk& chunk) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  buffer_queue_.push_back(chunk);
  if (buffer_queue_.size() > queue_size_) {
    buffer_queue_.pop_front();
    AWARN << "write bag buffer overflow";  
  }
  lock.unlock();
  cond_.notify_one();
}

void BagRecorder::WriteThread() {
  while(true) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cond_.wait(lock, [this]{return (!buffer_queue_.empty() || !writable_);});
    if (!writable_) {
      return;
    }
    auto chunk = buffer_queue_.front();
    buffer_queue_.pop_front();
    lock.unlock();
    std::lock_guard<std::mutex> guard(writer_mutex_);
    writer_->FeedData(chunk);
  }
}

}  // namespace message
}  // namespace common
}  // namespace roadstar