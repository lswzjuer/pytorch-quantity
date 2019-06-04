#include "modules/common/log.h"

namespace roadstar {
namespace common {

void InitLogging(const char *argv0) {
#ifdef USING_G3LOG
  g3::InitG3Logging(argv0);
#else
  google::InitGoogleLogging(argv0);
#endif  // USING_G3LOG
}

void SetStderrLoggingLevel(LogLevel level) {
#ifdef USING_G3LOG
  switch (level) {
    case FATAL_LEVEL:
      g3::SetStderrLogging(G3LOG_FATAL);
      break;
    case ERROR_LEVEL:
      g3::SetStderrLogging(G3LOG_ERROR);
      break;
    case WARNING_LEVEL:
      g3::SetStderrLogging(G3LOG_WARNING);
      break;
    case INFO_LEVEL:
      g3::SetStderrLogging(G3LOG_INFO);
      break;
    default:
      g3::SetStderrLogging(G3LOG_DEBUG);
      break;
  }
#else
  switch (level) {
    case FATAL_LEVEL:
      google::SetStderrLogging(google::GLOG_FATAL);
      break;
    case ERROR_LEVEL:
      google::SetStderrLogging(google::GLOG_ERROR);
      break;
    case WARNING_LEVEL:
      google::SetStderrLogging(google::GLOG_WARNING);
      break;
    case INFO_LEVEL:
      google::SetStderrLogging(google::GLOG_INFO);
      break;
    default:
      google::SetStderrLogging(google::GLOG_INFO);
      break;
  }
#endif
}

}  // namespace common
}  // namespace roadstar
