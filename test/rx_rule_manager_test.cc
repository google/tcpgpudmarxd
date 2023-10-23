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

#include <absl/container/flat_hash_map.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <linux/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_client.h"
#include "include/socket_helper.h"
#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.pb.h"
#include "proto/unix_socket_proto.pb.h"

namespace gpudirect_tcpxd {

using gpudirect_tcpxd::FlowSteerNtuple;
using gpudirect_tcpxd::GpuRxqConfigurationList;
using gpudirect_tcpxd::GpuRxqScaler;
using gpudirect_tcpxd::NicRulesBank;
using gpudirect_tcpxd::RxRuleManager;
using testing::_;
using testing::DoAll;
using testing::ElementsAre;
using testing::Pair;
using testing::Return;

class MockNicConfigurator : public gpudirect_tcpxd::NicConfiguratorInterface {
 public:
  MOCK_METHOD(absl::Status, Init, (), (override));
  MOCK_METHOD(void, Cleanup, (), (override));
  MOCK_METHOD(absl::Status, TogglePrivateFeature,
              (const std::string& ifname, const std::string& feature, bool on),
              (override));
  MOCK_METHOD(absl::Status, ToggleFeature,
              (const std::string& ifname, const std::string& feature, bool on),
              (override));
  MOCK_METHOD(absl::Status, SetRss, (const std::string& ifname, int num_queues),
              (override));
  MOCK_METHOD(absl::Status, AddFlow,
              (const std::string& ifname, const FlowSteerNtuple& ntuple,
               int queue_id, int location_id),
              (override));
  MOCK_METHOD(absl::Status, RemoveFlow,
              (const std::string& ifname, int location_id), (override));
  MOCK_METHOD(absl::Status, SetIpRoute,
              (const std::string& ifname, int min_rto, bool quickack),
              (override));
  MOCK_METHOD(absl::Status, RunSystem, (const std::string& command),
              (override));
  MOCK_METHOD(absl::StatusOr<__u32>, GetStat,
              ((const std::string& command), (const std::string& statname)),
              (override));
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
  auto* config = list.add_gpu_rxq_configs();
  auto* gpu_info = config->add_gpu_infos();
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
  EXPECT_EQ(rx_rule_manager.if_workers_.size(), 2);
  EXPECT_TRUE(rx_rule_manager.if_workers_.count("eth0"));
  EXPECT_TRUE(rx_rule_manager.if_workers_.count("eth1"));

  EXPECT_EQ(rx_rule_manager.ifname_to_first_gpu_map_.size(), 2);
  EXPECT_EQ(rx_rule_manager.ifname_to_first_gpu_map_["eth0"], "a.b.c.d");
  EXPECT_EQ(rx_rule_manager.ifname_to_first_gpu_map_["eth1"], "i.j.k.l");

  EXPECT_EQ(rx_rule_manager.gpu_to_ifname_map_["a.b.c.d"], "eth0");
  EXPECT_EQ(rx_rule_manager.gpu_to_ifname_map_["i.j.k.l"], "eth1");

  EXPECT_THAT(rx_rule_manager.GetQueueIds("a.b.c.d")->queue_ids(),
              ElementsAre(1));
  EXPECT_THAT(rx_rule_manager.GetQueueIds("r.s.t.u")->queue_ids(),
              ElementsAre(2));
  EXPECT_THAT(rx_rule_manager.GetQueueIds("i.j.k.l")->queue_ids(),
              ElementsAre(3));
}

class RxRuleManagerTest : public ::testing::Test {
 protected:
  const std::string kPrefix = "/tmp";

  MockNicConfigurator nic_configurator_;
  std::unique_ptr<RxRuleManager> rx_rule_manager_;
  GpuRxqConfigurationList configs_list_;

  static GpuRxqConfigurationList GetMultiNicConfig() {
    GpuRxqConfigurationList list;
    GpuRxqConfiguration* config;
    GpuInfo* gpu_info;

    // First NIC: eth0
    config = list.add_gpu_rxq_configs();
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("a.b.c.d");
    gpu_info->add_queue_ids(1);
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("e.f.g.h");
    gpu_info->add_queue_ids(2);
    config->set_nic_pci_addr("aa.bb.cc.dd");
    config->set_ifname("eth0");

    // Second NIC: eth1
    config = list.add_gpu_rxq_configs();
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("i.j.k.l");
    gpu_info->add_queue_ids(1);
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("m.n.o.p");
    gpu_info->add_queue_ids(2);
    config->set_nic_pci_addr("ee.ff.gg.hh");
    config->set_ifname("eth1");

    // Second NIC: eth2
    config = list.add_gpu_rxq_configs();
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("q.r.s.t");
    gpu_info->add_queue_ids(1);
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("u.v.w.x");
    gpu_info->add_queue_ids(2);
    config->set_nic_pci_addr("ii.jj.kk.ll");
    config->set_ifname("eth2");

    // Second NIC: eth2
    config = list.add_gpu_rxq_configs();
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("y.z.a.b");
    gpu_info->add_queue_ids(1);
    gpu_info = config->add_gpu_infos();
    gpu_info->set_gpu_pci_addr("c.d.e.f");
    gpu_info->add_queue_ids(2);
    config->set_nic_pci_addr("mm.nn.oo.pp");
    config->set_ifname("eth3");

    list.set_max_rx_rules(20000);
    return list;
  }

  void UpdateFlowSteerRule(const std::string& gpu_pci_addr,
                           gpudirect_tcpxd::FlowSteerRuleOp op,
                           const std::string& src_ip, const std::string& dst_ip,
                           uint16_t src_port, uint16_t dst_port,
                           bool check_ok = true) {
    RxRuleClient rx_rule_client(kPrefix);

    FlowSteerNtuple flow_steer_ntuple;
    flow_steer_ntuple.src_sin = gpudirect_tcpxd::AddressFromStr(src_ip).sin;
    flow_steer_ntuple.dst_sin = gpudirect_tcpxd::AddressFromStr(dst_ip).sin;
    gpudirect_tcpxd::SetAddressPort(
        (union SocketAddress*)&flow_steer_ntuple.src_sin, src_port);
    gpudirect_tcpxd::SetAddressPort(
        (union SocketAddress*)&flow_steer_ntuple.dst_sin, dst_port);
    flow_steer_ntuple.flow_type = 1;

    if (check_ok) {
      EXPECT_TRUE(rx_rule_client
                      .UpdateFlowSteerRule(op, flow_steer_ntuple, gpu_pci_addr)
                      .ok());
    } else {
      auto status = rx_rule_client.UpdateFlowSteerRule(op, flow_steer_ntuple,
                                                       gpu_pci_addr);
      EXPECT_FALSE(status.ok());
    }
  }

  void SetUp() override {
    configs_list_ = GetMultiNicConfig();
    rx_rule_manager_ = std::make_unique<RxRuleManager>(configs_list_, kPrefix,
                                                       &nic_configurator_);
    CHECK_EQ(rx_rule_manager_->Init(), absl::OkStatus());
  }
};

TEST_F(RxRuleManagerTest, ParallelInstallAndUninstallRules) {
  constexpr size_t kNumRulesPerGpu = 10;
  constexpr size_t kNumNic = 4;
  constexpr size_t kNumGpuPerNic = 2;

  constexpr size_t KTotalNumFlows = kNumNic * kNumGpuPerNic * kNumRulesPerGpu;
  constexpr size_t kTotalRulesPerNic = kNumGpuPerNic * kNumRulesPerGpu;
  constexpr absl::Duration kAddFlowLatency = absl::Milliseconds(5);

  // Introduce latency in the AddFlow call, and count the number of AddFlow()
  // per interface.
  absl::Mutex mu;
  absl::flat_hash_map<std::string, size_t> if_counters;
  EXPECT_CALL(nic_configurator_, AddFlow(_, _, _, _))
      .Times(KTotalNumFlows)
      .WillRepeatedly(DoAll(
          [&](const std::string& ifname, const FlowSteerNtuple& ntuple,
              int queue_id, int location_id) {
            absl::SleepFor(kAddFlowLatency);
            {
              absl::MutexLock lock(&mu);
              if_counters[ifname]++;
            }
          },
          Return(absl::OkStatus())));

  // Start a thread per GPU, and try to install rule.
  std::vector<std::thread> threads;
  threads.reserve(kNumNic * kNumGpuPerNic);
  absl::Time start = absl::Now();
  for (size_t nic_id = 0; nic_id < configs_list_.gpu_rxq_configs_size();
       nic_id++) {
    auto& gpu_rxq_config = configs_list_.gpu_rxq_configs().at(nic_id);
    for (size_t gpu_id = 0; gpu_id < gpu_rxq_config.gpu_infos_size();
         gpu_id++) {
      auto& gpu_info = gpu_rxq_config.gpu_infos().at(gpu_id);
      threads.push_back(std::thread([gpu_info, gpu_id, nic_id, this]() {
        for (int i = 0; i < kNumRulesPerGpu; i++) {
          UpdateFlowSteerRule(
              gpu_info.gpu_pci_addr(), CREATE,
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id + 1),
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id), 3000 + i,
              3000 + gpu_id * i);
        }
      }));
    }
  }

  // Wait until the rules are all installed.
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  absl::Time end = absl::Now();

  // Check the latency speedup. At least it should have 1/3 of the serial
  // latency.
  EXPECT_LT((end - start),
            absl::Duration(kAddFlowLatency * KTotalNumFlows) / 3);

  // Check the installed rules count.
  EXPECT_THAT(if_counters,
              ::testing::UnorderedElementsAre(Pair("eth0", kTotalRulesPerNic),
                                              Pair("eth1", kTotalRulesPerNic),
                                              Pair("eth2", kTotalRulesPerNic),
                                              Pair("eth3", kTotalRulesPerNic)));

  EXPECT_CALL(nic_configurator_, RemoveFlow(_, _))
      .Times(KTotalNumFlows)
      .WillRepeatedly(DoAll(
          [&](const std::string& ifname, int location_id) {
            absl::SleepFor(kAddFlowLatency);
            {
              absl::MutexLock lock(&mu);
              if_counters[ifname]--;
            }
          },
          Return(absl::OkStatus())));

  threads.clear();
  threads.reserve(kNumNic * kNumGpuPerNic);
  start = absl::Now();
  for (size_t nic_id = 0; nic_id < configs_list_.gpu_rxq_configs_size();
       nic_id++) {
    auto& gpu_rxq_config = configs_list_.gpu_rxq_configs().at(nic_id);
    for (size_t gpu_id = 0; gpu_id < gpu_rxq_config.gpu_infos_size();
         gpu_id++) {
      auto& gpu_info = gpu_rxq_config.gpu_infos().at(gpu_id);

      threads.push_back(std::thread([gpu_info, gpu_id, nic_id, this]() {
        for (int i = 0; i < kNumRulesPerGpu; i++) {
          UpdateFlowSteerRule(
              gpu_info.gpu_pci_addr(), DELETE,
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id + 1),
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id), 3000 + i,
              3000 + gpu_id * i);
        }
      }));
    }
  }

  // Wait until the rules are all removed.
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  end = absl::Now();

  // Check the latency speedup. At least it should have at least 1/3 of the
  // serial latency.
  EXPECT_LT((end - start),
            absl::Duration(kAddFlowLatency * KTotalNumFlows) / 3);

  // Check the rules count after uninstallation.
  EXPECT_THAT(if_counters, ::testing::UnorderedElementsAre(
                               Pair("eth0", 0), Pair("eth1", 0),
                               Pair("eth2", 0), Pair("eth3", 0)));
}

TEST_F(RxRuleManagerTest, UnKnownGpuAddr) {
  constexpr size_t kNumRulesPerGpu = 10;
  constexpr size_t kNumNic = 4;
  constexpr size_t kNumGpuPerNic = 2;

  constexpr size_t KTotalNumFlows = kNumNic * kNumGpuPerNic * kNumRulesPerGpu;
  constexpr size_t kTotalRulesPerNic = kNumGpuPerNic * kNumRulesPerGpu;
  constexpr absl::Duration kAddFlowLatency = absl::Milliseconds(5);

  // If gpu is unknown, the request should be early-exited
  EXPECT_CALL(nic_configurator_, AddFlow(_, _, _, _)).Times(0);

  // Start a thread per GPU, and try to install rule
  std::vector<std::thread> threads;
  threads.reserve(kNumNic * kNumGpuPerNic);
  for (size_t nic_id = 0; nic_id < configs_list_.gpu_rxq_configs_size();
       nic_id++) {
    auto& gpu_rxq_config = configs_list_.gpu_rxq_configs().at(nic_id);
    for (size_t gpu_id = 0; gpu_id < gpu_rxq_config.gpu_infos_size();
         gpu_id++) {
      auto& gpu_info = gpu_rxq_config.gpu_infos().at(gpu_id);
      threads.push_back(std::thread([gpu_info, gpu_id, nic_id, this]() {
        for (int i = 0; i < kNumRulesPerGpu; i++) {
          UpdateFlowSteerRule(
              "dummy", CREATE,
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id + 1),
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id), 3000 + i,
              3000 + gpu_id * i, false);
        }
      }));
    }
  }

  // Wait until the rules are all installed
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
}

TEST_F(RxRuleManagerTest, FailedAddFlow) {
  constexpr size_t kNumRulesPerGpu = 10;
  constexpr size_t kNumNic = 4;
  constexpr size_t kNumGpuPerNic = 2;

  constexpr size_t KTotalNumFlows = kNumNic * kNumGpuPerNic * kNumRulesPerGpu;
  constexpr size_t kTotalRulesPerNic = kNumGpuPerNic * kNumRulesPerGpu;
  constexpr absl::Duration kAddFlowLatency = absl::Milliseconds(5);

  // Failed on the AddFlow for each GPU
  EXPECT_CALL(nic_configurator_, AddFlow(_, _, _, _))
      .WillRepeatedly(Return(absl::InternalError("Mocked error")));

  // Start a thread per GPU, and try to install rule
  std::vector<std::thread> threads;
  threads.reserve(kNumNic * kNumGpuPerNic);
  for (size_t nic_id = 0; nic_id < configs_list_.gpu_rxq_configs_size();
       nic_id++) {
    auto& gpu_rxq_config = configs_list_.gpu_rxq_configs().at(nic_id);
    for (size_t gpu_id = 0; gpu_id < gpu_rxq_config.gpu_infos_size();
         gpu_id++) {
      auto& gpu_info = gpu_rxq_config.gpu_infos().at(gpu_id);
      threads.push_back(std::thread([gpu_info, gpu_id, nic_id, this]() {
        for (int i = 0; i < kNumRulesPerGpu; i++) {
          UpdateFlowSteerRule(
              gpu_info.gpu_pci_addr(), CREATE,
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id + 1),
              absl::StrFormat("192.168.%d.%d", gpu_id, nic_id), 3000 + i,
              3000 + gpu_id * i, false);
        }
      }));
    }
  }

  // Wait until the rules are all installed
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
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
