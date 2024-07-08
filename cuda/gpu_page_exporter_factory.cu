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

#include <memory>
#include <string>

#include "cuda/cu_ipc_memfd_exporter.cuh"
#include "cuda/cuda_ipc_memhandle_exporter.cuh"
#include "cuda/gpu_page_exporter_factory.cuh"
#include "include/gpu_page_exporter_interface.h"

namespace gpudirect_tcpxd {
std::unique_ptr<GpuPageExporterInterface> GpuPageExporterFactory::Build(
    const std::string& type) {
  if (type == "file") {
    return std::make_unique<CudaIpcMemhandleExporter>();
  } else if (type == "fd") {
    return std::make_unique<CuIpcMemfdExporter>();
  }
  return nullptr;
}
}  // namespace gpudirect_tcpxd
