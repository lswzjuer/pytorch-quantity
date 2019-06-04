#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_STATISTICAL_ANALYSIS_INTERFACE_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_STATISTICAL_ANALYSIS_INTERFACE_H


namespace roadstar {
namespace integration_test {


class StatisticalAnalysisInterface {
 public:
  virtual void Analyze() = 0;
  virtual ~StatisticalAnalysisInterface(){}
};

}  // namespace integration_test
}  // namespace roadstar

#endif
