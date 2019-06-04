#include "modules/common/log.h"
#include "modules/common/time/time.h"
#include "modules/common/util/concurrent_queue.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

namespace {
const int kMaxSize = 10000;
const int kSleepTime = 200;
const double kMaxAllowTimeDiff = 0.01;
}  // namespace

roadstar::common::ConcurrentQueue<std::string> concurrent_queue(kMaxSize);

void Writer(int sleep_for_time) {
  while (true) {
    std::string input_message =
        std::to_string(roadstar::common::time::Clock::NowInSecond()) +
        " -> Push";
    concurrent_queue.Push(input_message);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_for_time));
  }
}

void Reader() {
  while (true) {
    std::string out = concurrent_queue.Pop();
    double pop_time = roadstar::common::time::Clock::NowInSecond();
    double push_time = std::stod(out.substr(0, out.find_first_of(' ')));

    std::string current_pop_message = std::to_string(pop_time) + " -> Pop";
    double time_diff = pop_time - push_time;

    if (time_diff >= kMaxAllowTimeDiff) {
      AERROR << out << "  " << current_pop_message
             << "  Time diff: " << std::to_string(time_diff);
    } else {
      AINFO << out << "   " << current_pop_message
            << "  Time diff: " << std::to_string(time_diff);
    }
  }
}

int main(int argc, char **argv) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::thread writer(Writer, kSleepTime);
  std::thread reader(Reader);

  writer.join();
  reader.join();

  return 0;
}
