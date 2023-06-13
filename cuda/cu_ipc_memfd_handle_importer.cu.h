#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_CU_IPC_MEMFD_HANDLE_IMPORTER_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_CU_IPC_MEMFD_HANDLE_IMPORTER_H_

#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/cuda/gpu_page_handle_interface.cu.h"
#include "third_party/absl/status/statusor.h"

namespace tcpdirect {

class CuIpcMemfdHandleImporter {
 public:
  static absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>> Import(
      const std::string& prefix, const std::string& gpu_pci_addr);
};

}  // namespace tcpdirect

#endif
