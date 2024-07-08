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

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_GPU_PAGE_HANDLE_INTERFACE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_GPU_PAGE_HANDLE_INTERFACE_H_

#include <cuda.h>

namespace gpudirect_tcpxd {
class GpuPageHandleInterface {
 public:
  virtual ~GpuPageHandleInterface() = default;
  virtual CUdeviceptr GetGpuMem() = 0;
};
}  // namespace gpudirect_tcpxd
#endif
