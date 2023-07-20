// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/monstertruck_gpu_rxq_configurator.h"

#include <memory>
#include <utility>
#include <vector>

namespace gpudirect_tcpxd {
GpuRxqConfigurationList MonstertruckGpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    GpuInfo* gpu_info = configuration->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("0000:62:00.0");
    for (int i = 0; i < 15; ++i) {
      gpu_info->add_queue_ids(i);
    }
    configuration->set_nic_pci_addr("0000:63:00.0");
    configuration->set_ifname("hpn2");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    GpuInfo* gpu_info = configuration->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("0000:6a:00.0");
    for (int i = 0; i < 15; ++i) {
      gpu_info->add_queue_ids(i);
    }
    configuration->set_nic_pci_addr("0000:6b:00.0");
    configuration->set_ifname("hpn1");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    GpuInfo* gpu_info = configuration->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("0000:b3:00.0");
    for (int i = 0; i < 15; ++i) {
      gpu_info->add_queue_ids(i);
    }
    configuration->set_nic_pci_addr("0000:b4:00.0");
    configuration->set_ifname("hpn3");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    GpuInfo* gpu_info = configuration->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("0000:bb:00.0");
    for (int i = 0; i < 15; ++i) {
      gpu_info->add_queue_ids(i);
    }
    configuration->set_nic_pci_addr("0000:bc:00.0");
    configuration->set_ifname("hpn4");
  }
  config_list.set_tcpd_queue_size(14);
  config_list.set_rss_set_size(0);
  return config_list;
}
}  // namespace gpudirect_tcpxd
