#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_GPU_PAGE_EXPORTER_FACTORY_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_GPU_PAGE_EXPORTER_FACTORY_H_

#include <memory>
#include <string>

#include "include/gpu_page_exporter_interface.h"

namespace gpudirect_tcpxd {

class GpuPageExporterFactory {
 public:
  static std::unique_ptr<GpuPageExporterInterface> Build(
      const std::string& type);
};
}  // namespace gpudirect_tcpxd
#endif
