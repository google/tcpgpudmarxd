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

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_IPC_MEMHANDLE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_IPC_MEMHANDLE_H_

#include <cuda.h>

#include <string>

#include "cuda/gpu_page_handle_interface.cuh"

namespace gpudirect_tcpxd {
class CudaIpcMemhandle : public GpuPageHandleInterface {
 public:
  CudaIpcMemhandle(const std::string& handle);
  ~CudaIpcMemhandle() override;
  CUdeviceptr GetGpuMem() override { return ptr_; }

 private:
  CUipcMemHandle mem_handle_;
  CUdeviceptr ptr_;
};
}  // namespace gpudirect_tcpxd

#endif
