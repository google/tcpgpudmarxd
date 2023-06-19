#include "include/nic_configurator_factory.h"

#include <memory>
#include <string>

#include "include/ethtool_nic_configurator.h"
#include "include/ethtool_no_headersplit_nic_configurator.h"

namespace tcpdirect {
std::unique_ptr<NicConfiguratorInterface> NicConfiguratorFactory::Build(
    const std::string& name) {
  if (name == "monstertruck" || name == "predvt") {
    return std::make_unique<EthtoolNoHeaderSplitNicConfigurator>();
  }
  return std::make_unique<EthtoolNicConfigurator>();
}
}  // namespace tcpdirect
