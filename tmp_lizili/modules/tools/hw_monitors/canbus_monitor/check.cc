#include <iostream>

#include "canlib.h"
#include "modules/msgs/dreamview/proto/hw_monitor_result.pb.h"

using roadstar::dreamview::HWMonitorResult;

HWMonitorResult Check() {
  HWMonitorResult result;

  canInitializeLibrary();

  int chan_count;
  auto stat = canGetNumberOfChannels(&chan_count);
  if (stat != canOK) {
    char buf[50];
    buf[0] = '\0';
    canGetErrorText(stat, buf, sizeof(buf));

    result.set_status(HWMonitorResult::FAILED);
    result.set_error_message("canGetNumberOfChannels: failed, stat=" +
                             std::to_string(stat) + " (" + buf + ")");
  } else {
    if (chan_count > 0) {
      result.set_status(HWMonitorResult::OK);
    } else {
      result.set_status(HWMonitorResult::FAILED);
      result.set_error_message("no valid channel");
    }
  }
  return result;
}

int main() {
  std::cout << Check().SerializeAsString() << std::flush;
  return 0;
}
