/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <linux/netdevice.h>
#include <linux/netlink.h>
#include <linux/types.h>
#include <pthread.h>
#include <sys/socket.h>

#include <memory>

#include "include/ethtool_nic_configurator.h"

#define NIC_COUNT 4

namespace gpudirect_tcpxd {

absl::StatusOr<__u32> read_reset_cnt(NicConfiguratorInterface* nic_configurator,
                                     const std::string& ifname);

struct netlink_thread {
  pthread_t thread_id;
  /* all values should be initialized to 0xFFFFFFFF */
  __u32 reset_cnts[NIC_COUNT];
  NicConfiguratorInterface* nic_configurator;
};

bool CheckVFReset(struct netlink_thread* nt, std::string ifname,
                  int reset_cnt_idx);
}  // namespace gpudirect_tcpxd
