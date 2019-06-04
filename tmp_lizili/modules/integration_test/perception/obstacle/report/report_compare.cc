#include <fstream>
#include <iostream>
#include <string>
#include "modules/common/log.h"
#include "modules/integration_test/perception/obstacle/model/report_model.h"

using ReportModel = roadstar::integration_test::ReportModel;

int main(int argc, char** argv) {
  if (argc == 1) {
    AERROR << "please input the report xml path of bag file parameters"
           << std::endl;
    return 0;
  }
  std::string file = argv[1];
  std::ifstream fin(file);
  std::string self;
  std::string other;
  std::string diff;
  fin >> self;
  fin >> other;
  fin >> diff;
  std::cout << "self = " << self.c_str() << std::endl
            << " others = " << other.c_str() << std::endl
            << " diff = " << diff.c_str();
  ReportModel self_model;
  ReportModel other_model;
  self_model.ParseFromFile(self);
  other_model.ParseFromFile(other);
  self_model.Diff(&other_model, diff);
  return 0;
}
