#include "include/dummy_ethtool_nic_configurator.h"

#include <string>

#include <absl/log/log.h>

namespace tcpdirect {

absl::Status DummyEthtoolNicConfigurator::RunSystem(
    const std::string& command) {
  LOG(INFO) << "Run system: " << command;
  return absl::OkStatus();
}
}  // namespace tcpdirect
