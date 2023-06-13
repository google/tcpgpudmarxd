#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_COMMON_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_COMMON_H_

#include <unordered_map>

#include "base/logging.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace tcpdirect {

#define CU_ASSERT_SUCCESS(expr)                                         \
  {                                                                     \
    {                                                                   \
      CUresult __err = (expr);                                          \
      if (__err != CUDA_SUCCESS) {                                      \
        const char* name = "[unknown]";                                 \
        const char* reason = "[unknown]";                               \
        if (cuGetErrorName(__err, &name)) {                             \
          LOG(ERROR) << "Error: error getting error name";              \
        }                                                               \
        if (cuGetErrorString(__err, &reason)) {                         \
          LOG(ERROR) << "Error: error getting error string";            \
        }                                                               \
        LOG(FATAL) << absl::StrFormat(                                  \
            "cuda error detected! name: %s; string: %s", name, reason); \
      }                                                                 \
    }                                                                   \
  }

#define CUDA_ASSERT_SUCCESS(expr)                                       \
  {                                                                     \
    {                                                                   \
      cudaError_t __err = (expr);                                       \
      if (__err != cudaSuccess) {                                       \
        const char* name = cudaGetErrorName(__err);                     \
        const char* reason = cudaGetErrorString(__err);                 \
        LOG(FATAL) << absl::StrFormat(                                  \
            "cuda error detected! name: %s; string: %s", name, reason); \
      }                                                                 \
    }                                                                   \
  }

using PciAddrToGpuIdxMap = std::unordered_map<std::string, int>;
void GetPciAddrToGpuIndexMap(PciAddrToGpuIdxMap* pciaddr_gpuidx_map);

}  // namespace tcpdirect
#endif
