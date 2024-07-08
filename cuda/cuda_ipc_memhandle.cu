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

#include <absl/log/log.h>

#include "cuda/common.cuh"
#include "cuda/cuda_ipc_memhandle.cuh"

namespace gpudirect_tcpxd {

CudaIpcMemhandle::CudaIpcMemhandle(const std::string& handle) {
  memcpy(&mem_handle_, handle.data(), handle.size());
  CU_ASSERT_SUCCESS(
      cuIpcOpenMemHandle(&ptr_, mem_handle_, cudaIpcMemLazyEnablePeerAccess));
}

CudaIpcMemhandle::~CudaIpcMemhandle() { cuIpcCloseMemHandle(ptr_); }

}  // namespace gpudirect_tcpxd
