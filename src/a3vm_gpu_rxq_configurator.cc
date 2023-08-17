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

#include "include/a3vm_gpu_rxq_configurator.h"

#include <memory>
#include <utility>
#include <vector>

namespace gpudirect_tcpxd {
namespace {}
GpuRxqConfigurationList A3VmGpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:04:00.0");
      gpu_info->add_queue_ids(8);
      gpu_info->add_queue_ids(9);
      gpu_info->add_queue_ids(10);
      gpu_info->add_queue_ids(11);
    }
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:05:00.0");
      gpu_info->add_queue_ids(12);
      gpu_info->add_queue_ids(13);
      gpu_info->add_queue_ids(14);
      gpu_info->add_queue_ids(15);
    }
    configuration->set_nic_pci_addr("0000:06:00.0");
    configuration->set_ifname("eth1");
  }
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:0a:00.0");
      gpu_info->add_queue_ids(8);
      gpu_info->add_queue_ids(9);
      gpu_info->add_queue_ids(10);
      gpu_info->add_queue_ids(11);
    }
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:0b:00.0");
      gpu_info->add_queue_ids(12);
      gpu_info->add_queue_ids(13);
      gpu_info->add_queue_ids(14);
      gpu_info->add_queue_ids(15);
    }
    configuration->set_nic_pci_addr("0000:0c:00.0");
    configuration->set_ifname("eth2");
  }
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:84:00.0");
      gpu_info->add_queue_ids(8);
      gpu_info->add_queue_ids(9);
      gpu_info->add_queue_ids(10);
      gpu_info->add_queue_ids(11);
    }
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:85:00.0");
      gpu_info->add_queue_ids(12);
      gpu_info->add_queue_ids(13);
      gpu_info->add_queue_ids(14);
      gpu_info->add_queue_ids(15);
    }
    configuration->set_nic_pci_addr("0000:86:00.0");
    configuration->set_ifname("eth3");
  }
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:8a:00.0");
      gpu_info->add_queue_ids(8);
      gpu_info->add_queue_ids(9);
      gpu_info->add_queue_ids(10);
      gpu_info->add_queue_ids(11);
    }
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:8b:00.0");
      gpu_info->add_queue_ids(12);
      gpu_info->add_queue_ids(13);
      gpu_info->add_queue_ids(14);
      gpu_info->add_queue_ids(15);
    }
    configuration->set_nic_pci_addr("0000:8c:00.0");
    configuration->set_ifname("eth4");
  }
  config_list.set_rss_set_size(8);
  config_list.set_tcpd_queue_size(8);
  config_list.set_max_rx_rules(100000);
  return config_list;
}

GpuRxqConfigurationList A3VmGpuRxqConfigurator4GPU4NIC::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:04:00.0");
      for (int i = 8; i < 16; ++i) {
        gpu_info->add_queue_ids(i);
      }
    }
    configuration->set_nic_pci_addr("0000:06:00.0");
    configuration->set_ifname("eth1");
  }
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:0a:00.0");
      for (int i = 8; i < 16; ++i) {
        gpu_info->add_queue_ids(i);
      }
    }
    configuration->set_nic_pci_addr("0000:0c:00.0");
    configuration->set_ifname("eth2");
  }
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:84:00.0");
      for (int i = 8; i < 16; ++i) {
        gpu_info->add_queue_ids(i);
      }
    }
    configuration->set_nic_pci_addr("0000:86:00.0");
    configuration->set_ifname("eth3");
  }
  {
    GpuRxqConfiguration *configuration = config_list.add_gpu_rxq_configs();
    {
      GpuInfo *gpu_info = configuration->add_gpu_infos();
      gpu_info->set_gpu_pci_addr("0000:8a:00.0");
      for (int i = 8; i < 16; ++i) {
        gpu_info->add_queue_ids(i);
      }
    }
    configuration->set_nic_pci_addr("0000:8c:00.0");
    configuration->set_ifname("eth4");
  }
  config_list.set_rss_set_size(8);
  config_list.set_tcpd_queue_size(8);
  config_list.set_max_rx_rules(100000);
  return config_list;
}
}  // namespace gpudirect_tcpxd
