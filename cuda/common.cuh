/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_COMMON_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_COMMON_H_

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <cuda.h>

#include <unordered_map>

namespace gpudirect_tcpxd {

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

}  // namespace gpudirect_tcpxd
#endif
