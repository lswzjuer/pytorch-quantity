#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "modules/common/log.h"
#include "modules/integration_test/common/xml_param/xml_param_reader.h"

const char kBagsName[] = "bags_name";

void Out(const std::string &str) {
  fprintf(stdout, "%s", str.c_str());
}

int main(int argc, char **argv) {
  if (argc < 3) {
    AERROR
        << "please input the cmd and the config xml path of bag file parameters"
        << std::endl;
    std::string str("");
    Out(str);
    return 0;
  }
  std::string xml_path(argv[2]);
  roadstar::integration_test::XMLParamReader reader(xml_path);
  if (!reader.IsSucceedToLoad()) {
    AERROR << "load xml config file \"" << xml_path.c_str()
           << "\" failed. exiting now...";
    std::string str("");
    Out(str);
    return 0;
  }
  std::string key(argv[1]);
  if (key.length() > 0) {
    std::string value;
    if (key == kBagsName) {
      value = reader.GetBagsNameString();
    } else {
      value = reader.GetValueViaKey(key);
    }
    Out(value);
  }
  return 0;
}
