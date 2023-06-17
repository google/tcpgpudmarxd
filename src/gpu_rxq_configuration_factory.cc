#include "include/gpu_rxq_configuration_factory.h"

#include <absl/log/log.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "include/a3_gpu_rxq_configurator.cuh"
#include "include/a3vm_gpu_rxq_configurator.h"
#include "include/gpu_rxq_configurator_interface.h"
#include "include/monstertruck_gpu_rxq_configurator.h"
#include "include/predvt_gpu_rxq_configurator.h"
#include "proto/gpu_rxq_configuration.pb.h"

namespace tcpdirect {

namespace {
using google::protobuf::TextFormat;
bool GetConfigurationList(std::string filename,
                          GpuRxqConfigurationList* configuration_list) {
  std::ifstream text_proto_filestream(filename);
  std::stringstream buffer;
  buffer << text_proto_filestream.rdbuf();

  TextFormat::Parser parser;
  const bool success = parser.ParseFromString(buffer.str(), configuration_list);
  return success;
}
}  // namespace

GpuRxqConfigurationList GpuRxqConfigurationFactory::BuildPreset(
    const std::string& name) {
  std::unique_ptr<GpuRxqConfiguratorInterface> configurator;
  if (name == "monstertruck") {
    configurator = std::make_unique<MonstertruckGpuRxqConfigurator>();
  } else if (name == "predvt") {
    configurator = std::make_unique<PreDvtGpuRxqConfigurator>();
  } else if (name == "a3vm") {
    configurator = std::make_unique<A3VmGpuRxqConfigurator>();
  } else if (name == "a3vm4gpu4nic") {
    configurator = std::make_unique<A3VmGpuRxqConfigurator4GPU4NIC>();
  } else {  // auto
    configurator = std::make_unique<A3GpuRxqConfigurator>();
  }
  return configurator->GetConfigurations();
}

GpuRxqConfigurationList GpuRxqConfigurationFactory::FromFile(
    const std::string& filename) {
  GpuRxqConfigurationList gpu_configuration_list;
  if (!GetConfigurationList(filename, &gpu_configuration_list)) {
    LOG(ERROR) << "Unable to parse textproto from " << filename;
  }
  return gpu_configuration_list;
}

GpuRxqConfigurationList GpuRxqConfigurationFactory::FromCmdLine(
    const std::string& proto_string) {
  GpuRxqConfigurationList gpu_configuration_list;
  TextFormat::Parser parser;
  const bool success =
      parser.ParseFromString(proto_string, &gpu_configuration_list);
  if (!success) {
    LOG(ERROR) << "Unable to parse textproto from " << proto_string;
  }
  return gpu_configuration_list;
}
}  // namespace tcpdirect
