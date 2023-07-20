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

#include "include/rx_rule_client.h"

#include <absl/functional/bind_front.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>
#include <sys/un.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "include/flow_steer_ntuple.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_message.pb.h"
#include "gmock/gmock.h"
#include <google/protobuf/text_format.h>

namespace {
using gpudirect_tcpxd::GpuRxqConfigurationList;

TEST(GpuRxqConfigurationFactoryTest, GetFromFileSuccess) {
  GpuRxqConfigurationList list;
  auto *config = list.add_gpu_rxq_configs();
  auto *gpu_info = config->add_gpu_infos();
  gpu_info->set_gpu_pci_addr("0000:62:00.0");
  gpu_info->add_queue_ids(0);
  config->set_nic_pci_addr("0000:63:00.0");
  config->set_ifname("hpn2");
  std::string filename =
      "../../test/data/gpu_rxq_configuration_factory_test.txt";
  auto gpu_rxq_configs =
      gpudirect_tcpxd::GpuRxqConfigurationFactory::FromFile(filename);
  EXPECT_EQ(gpu_rxq_configs.DebugString(), list.DebugString());
}

TEST(GpuRxqConfigurationFactoryTest, GetFromCmdLineSuccess) {
  GpuRxqConfigurationList list;
  std::string proto_string = R"pb(
    gpu_rxq_configs {
    gpu_infos {
        gpu_pci_addr: "0000:62:00.0"
        queue_ids: 0
    }
        nic_pci_addr: "0000:63:00.0"
        ifname: "hpn2";
    }
  )pb";
  list =
      gpudirect_tcpxd::GpuRxqConfigurationFactory::FromCmdLine(proto_string);
  EXPECT_EQ(list.DebugString(), list.DebugString());
}
} // namespace