#include "experimental/users/chechenglin/tcpgpudmad/include/nic_configurator_factory.h"

#include <memory>
#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/ethtool_nic_configurator.h"
#include "experimental/users/chechenglin/tcpgpudmad/include/ethtool_no_headersplit_nic_configurator.h"
#include "experimental/users/chechenglin/tcpgpudmad/include/ioctl_nic_configurator.h"

namespace tcpdirect {
std::unique_ptr<NicConfiguratorInterface> NicConfiguratorFactory::Build(
    const std::string& name) {
  if (name == "ioctl") {
    return std::make_unique<IoctlNicConfigurator>();
  } else if (name == "monstertruck" || name == "predvt") {
    return std::make_unique<EthtoolNoHeaderSplitNicConfigurator>();
  } else if (name == "a3vm") {
    return std::make_unique<EthtoolNicConfigurator>();
  }
  return nullptr;
}
}  // namespace tcpdirect
