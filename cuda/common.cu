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

#include "cuda/common.cuh"

namespace gpudirect_tcpxd {

void GetPciAddrToGpuIndexMap(PciAddrToGpuIdxMap* pci_addr_to_gpu_idx) {
  int num_gpu = 0;
  CUDA_ASSERT_SUCCESS(cudaGetDeviceCount(&num_gpu));
  for (int i = 0; i < num_gpu; i++) {
    cudaDeviceProp prop;
    CUDA_ASSERT_SUCCESS(cudaGetDeviceProperties(&prop, i));
    std::string gpu_pci_addr = absl::StrFormat(
        "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    (*pci_addr_to_gpu_idx)[gpu_pci_addr] = i;
  }

  for (const auto& kv : *pci_addr_to_gpu_idx) {
    LOG(INFO) << absl::StrFormat("%s --> CUDA device %d", kv.first, kv.second);
  }
}
}  // namespace gpudirect_tcpxd
