#ifndef __EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_IPC_GPUMEM_FD_METADATA_H_
#define __EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_IPC_GPUMEM_FD_METADATA_H_
#include <stddef.h>

namespace tcpdirect {
struct IpcGpuMemFdMetadata {
  int fd{-1};
  size_t size{0};
  size_t align{0};
};
}  // namespace tcpdirect
#endif
