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
