#include "cuda/gpu_page_exporter_factory.cu.h"

#include <memory>
#include <string>

#include "cuda/cu_ipc_memfd_exporter.cu.h"
#include "cuda/cuda_ipc_memhandle_exporter.cu.h"
#include "include/gpu_page_exporter_interface.h"

namespace tcpdirect {
std::unique_ptr<GpuPageExporterInterface> GpuPageExporterFactory::Build(
    const std::string& type) {
  if (type == "file") {
    return std::make_unique<CudaIpcMemhandleExporter>();
  } else if (type == "fd") {
    return std::make_unique<CuIpcMemfdExporter>();
  }
  return nullptr;
}
}  // namespace tcpdirect
