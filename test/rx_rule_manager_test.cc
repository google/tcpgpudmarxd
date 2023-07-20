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

#include "include/rx_rule_manager.h"

#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {

using gpudirect_tcpxd::FlowSteerNtuple;
using gpudirect_tcpxd::GpuRxqConfigurationList;
using gpudirect_tcpxd::GpuRxqScaler;
using gpudirect_tcpxd::NicRulesBank;
using gpudirect_tcpxd::RxRuleManager;

class MockNicConfigurator : public gpudirect_tcpxd::NicConfiguratorInterface {
 public:
  MOCK_METHOD(absl::Status, Init, (), (override));
  MOCK_METHOD(void, Cleanup, (), (override));
  MOCK_METHOD(absl::Status, TogglePrivateFeature,
              (const std::string &ifname, const std::string &feature, bool on),
              (override));
  MOCK_METHOD(absl::Status, ToggleFeature,
              (const std::string &ifname, const std::string &feature, bool on),
              (override));
  MOCK_METHOD(absl::Status, SetRss, (const std::string &ifname, int num_queues),
              (override));
  MOCK_METHOD(absl::Status, AddFlow,
              (const std::string &ifname, const FlowSteerNtuple &ntuple,
               int queue_id, int location_id),
              (override));
  MOCK_METHOD(absl::Status, RemoveFlow,
              (const std::string &ifname, int location_id), (override));
};

TEST(RxRuleManagerInitTest, InitSuccess) {
  MockNicConfigurator nic_configurator;
  GpuRxqConfigurationList list;
  std::string prefix = "/tmp";
  RxRuleManager rx_rule_manager(list, prefix, &nic_configurator);
  EXPECT_EQ(rx_rule_manager.Init(), absl::OkStatus());
}

TEST(RxRuleManagerInitTest, PopBackSuccess) {
  MockNicConfigurator nic_configurator;
  GpuRxqConfigurationList list;
  std::string prefix = "/tmp/";
  RxRuleManager rx_rule_manager(list, prefix, &nic_configurator);
  EXPECT_EQ(rx_rule_manager.Init(), absl::OkStatus());
}

TEST(RxRuleManagerConstructorTest, InitGpuRxqConfigSuccess) {
  MockNicConfigurator nic_configurator;
  GpuRxqConfigurationList list;
  auto *config = list.add_gpu_rxq_configs();
  auto *gpu_info = config->add_gpu_infos();
  gpu_info->set_gpu_pci_addr("a.b.c.d");
  gpu_info->add_queue_ids(1);
  gpu_info = config->add_gpu_infos();
  gpu_info->set_gpu_pci_addr("r.s.t.u");
  gpu_info->add_queue_ids(2);
  config->set_nic_pci_addr("e.f.g.h");
  config->set_ifname("eth0");

  config = list.add_gpu_rxq_configs();
  gpu_info = config->add_gpu_infos();
  gpu_info->set_gpu_pci_addr("i.j.k.l");
  gpu_info->add_queue_ids(3);
  config->set_nic_pci_addr("m.n.o.p");
  config->set_ifname("eth1");

  std::string prefix = "/tmp";
  RxRuleManager rx_rule_manager(list, prefix, &nic_configurator);
  EXPECT_EQ(rx_rule_manager.ifname_to_rules_bank_map_.size(), 2);
  EXPECT_TRUE(rx_rule_manager.ifname_to_rules_bank_map_.count("eth0"));
  EXPECT_TRUE(rx_rule_manager.ifname_to_rules_bank_map_.count("eth1"));

  EXPECT_EQ(rx_rule_manager.ifname_to_first_gpu_map_.size(), 2);
  EXPECT_EQ(rx_rule_manager.ifname_to_first_gpu_map_["eth0"], "a.b.c.d");
  EXPECT_EQ(rx_rule_manager.ifname_to_first_gpu_map_["eth1"], "i.j.k.l");

  EXPECT_EQ(rx_rule_manager.gpu_to_ifname_map_["a.b.c.d"], "eth0");
  EXPECT_EQ(rx_rule_manager.gpu_to_ifname_map_["i.j.k.l"], "eth1");

  EXPECT_TRUE(rx_rule_manager.gpu_to_rxq_scaler_map_.count("a.b.c.d"));
  EXPECT_TRUE(rx_rule_manager.gpu_to_rxq_scaler_map_.count("i.j.k.l"));
}

TEST(NicRulesBankTest, IsAvailble) {
  std::string ifname = "tmp";
  int max_rx_rules = 2;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  EXPECT_TRUE(nic_rules_bank.Availble());
}

TEST(NicRulesBankTest, RuleDoesExist) {
  std::string ifname = "tmp";
  int max_rx_rules = 2;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  size_t flow_hash = 1;
  nic_rules_bank.UseRule(flow_hash);
  EXPECT_TRUE(nic_rules_bank.RuleExists(flow_hash));
}

TEST(NicRulesBankTest, UseRuleSuccess) {
  std::string ifname = "tmp";
  int max_rx_rules = 2;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  size_t flow_hash = 1;
  int location_id = 0;
  EXPECT_EQ(location_id, nic_rules_bank.UseRule(flow_hash));
}

TEST(NicRulesBankTest, UseRuleFail) {
  std::string ifname = "tmp";
  int max_rx_rules = 0;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  size_t flow_hash = 1;
  EXPECT_EQ(nic_rules_bank.UseRule(flow_hash), -1);
}

TEST(NicRulesBankTest, ReturnRuleSuccess) {
  std::string ifname = "tmp";
  int max_rx_rules = 2;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  size_t flow_hash = 1;
  auto location_id = nic_rules_bank.UseRule(flow_hash);
  EXPECT_EQ(location_id, nic_rules_bank.ReturnRule(flow_hash));
}

TEST(NicRulesBankTest, ReturnRuleFail) {
  std::string ifname = "tmp";
  int max_rx_rules = 2;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  size_t flow_hash = 1;
  EXPECT_EQ(nic_rules_bank.ReturnRule(flow_hash), -1);
}

TEST(NicRulesBankTest, CheckLocationsInUse) {
  std::string ifname = "tmp";
  int max_rx_rules = 2;
  NicRulesBank nic_rules_bank(ifname, max_rx_rules);
  size_t flow_hash = 1;
  auto location_id = nic_rules_bank.UseRule(flow_hash);
  auto location_ids = nic_rules_bank.LocationsInUse();
  EXPECT_TRUE(
      std::count(location_ids.begin(), location_ids.end(), location_id));
}

TEST(GpuRxqScalerTest, HasValidQueueId) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  int queue_id = 0;
  EXPECT_TRUE(gpu_rxq_scaler.valid_queue_id(queue_id));
}

TEST(GpuRxqScalerTest, IsValid) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  int queue_id = 0;
  EXPECT_TRUE(gpu_rxq_scaler.valid());
}

TEST(GpuRxqScalerTest, AddFlowWithQueueIdSuccess) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  int queue_id = 0;
  int location_id = 0;
  gpu_rxq_scaler.AddFlow(location_id, queue_id);
  EXPECT_EQ(gpu_rxq_scaler.qid_qinfo_map_[queue_id].flow_counts, 1);
}

TEST(GpuRxqScalerTest, AddFlowWithoutQueueIdSuccess) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  int location_id = 0;

  // add flows to queue 0 & queue 1 and expect the next flow will be added to
  // queue 2
  gpu_rxq_scaler.AddFlow(location_id, 0);
  gpu_rxq_scaler.AddFlow(location_id, 1);
  auto queue_id = gpu_rxq_scaler.AddFlow(location_id);
  EXPECT_EQ(2, queue_id);
  EXPECT_EQ(gpu_rxq_scaler.qid_qinfo_map_[queue_id].flow_counts, 1);
}

TEST(GpuRxqScalerTest, RemoveFlowSuccess) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  int queue_id = 0;
  int location_id = 0;
  gpu_rxq_scaler.AddFlow(location_id, queue_id);
  auto before_size = gpu_rxq_scaler.location_to_queue_map_.size();
  EXPECT_EQ(gpu_rxq_scaler.qid_qinfo_map_[queue_id].flow_counts, 1);
  gpu_rxq_scaler.RemoveFlow(location_id);
  auto after_size = gpu_rxq_scaler.location_to_queue_map_.size();
  EXPECT_EQ(gpu_rxq_scaler.qid_qinfo_map_[queue_id].flow_counts, 0);
  EXPECT_GE(before_size, after_size);
}

TEST(GpuRxqScalerTest, RemoveFlowFail) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  int queue_id = 0;
  int location_id = 0;
  auto before_size = gpu_rxq_scaler.location_to_queue_map_.size();
  gpu_rxq_scaler.RemoveFlow(location_id);
  auto after_size = gpu_rxq_scaler.location_to_queue_map_.size();
  EXPECT_EQ(before_size, after_size);
}

TEST(GpuRxqScalerTest, PushQueueIdsSuccess) {
  std::string gpu_pci_addr = "addr";
  std::vector<int> queue_ids = {0, 1, 2};
  GpuRxqScaler gpu_rxq_scaler(gpu_pci_addr, queue_ids);
  std::vector<int> qids = {0, 1, 2};
  EXPECT_THAT(gpu_rxq_scaler.QueueIds(),
              testing::UnorderedElementsAre(0, 1, 2));

  // Add a flow to a new queue, and check that we can get the expected queue ids
  gpu_rxq_scaler.AddFlow(0, 3);
  EXPECT_THAT(gpu_rxq_scaler.QueueIds(),
              testing::UnorderedElementsAre(0, 1, 2, 3));
}

}  // namespace gpudirect_tcpxd
