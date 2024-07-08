/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include "include/vf_reset_detector.h"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <linux/if.h>
#include <linux/netdevice.h>
#include <linux/rtnetlink.h>
#include <stdio.h>

#include "include/nic_configurator_interface.h"

namespace gpudirect_tcpxd {

absl::StatusOr<__u32> read_reset_cnt(NicConfiguratorInterface* nic_configurator,
                                     const std::string& ifname) {
  std::string reset_cnt = "reset_cnt";

  return nic_configurator->GetStat(ifname, reset_cnt);
}

// Check if Virtual Function reset occurred:
// We assume VF reset has occurred if reset_cnt has changed
//
// return: true if VF reset was detected
bool CheckVFReset(struct netlink_thread* nt, std::string ifname,
                  int reset_cnt_idx) {
  int reset_cnt;
  absl::StatusOr<__u32> new_reset_cnt;

  reset_cnt = nt->reset_cnts[reset_cnt_idx];

  new_reset_cnt = read_reset_cnt(nt->nic_configurator, ifname);

  if (!new_reset_cnt.ok()) {
    LOG(ERROR) << absl::StrFormat("unable to read reset_cnt from %s", ifname);
    return false;
  }

  if (reset_cnt < *new_reset_cnt) {
    LOG(WARNING) << absl::StrFormat(
        "detected VF Reset: NIC %s, reset_cnt %u, new_reset_cnt %u", ifname,
        reset_cnt, *new_reset_cnt);
    nt->reset_cnts[reset_cnt_idx] = *new_reset_cnt;
    return true;
  }
  return false;
}
} /* namespace gpudirect_tcpxd */
