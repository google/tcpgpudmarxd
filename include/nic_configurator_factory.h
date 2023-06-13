#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_FACTORY_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_FACTORY_

#include <memory>
#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/nic_configurator_interface.h"

namespace tcpdirect {
class NicConfiguratorFactory {
 public:
  static std::unique_ptr<NicConfiguratorInterface> Build(
      const std::string& name);
};
}  // namespace tcpdirect
#endif
