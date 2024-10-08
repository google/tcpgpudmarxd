#!/bin/bash

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source "/tcpgpudmarxd/build/a3-tuning-scripts/sysctl_helper.sh"
GLOBAL_NETBASE_ERROR_COUNTER=0

# Print commands we're executing, and exit if any commands fail:
set -x

reset_route_param() {
  while read -r line; do
    # Note that suffix strings are removed in the reverse of the
    # order in which they are added:
    newline="${line%quickack*}"
    newline="${newline%rto_min*}"
    newline="${newline%initcwnd*}"
    ip route del ${newline} 2> /dev/null
    ret=$?
    if [[ $ret -ne 0 ]]; then
      ip route change ${newline}
    else
      ip route add ${newline}
    fi
  done < <(ip route show)
  ip route show
}

main() {
  reset_route_param

  SYSFS="/hostsysfs"
  PROCSYSFS="/hostprocsysfs"

  if [[ -d $SYSFS && -d $PROCSYSFS ]]; then
    echo "Use mounted '$PROCSYSFS' and '$SYSFS' ."
  else
    PROCSYSFS="/proc/sys"
    SYSFS="/sys"
    echo "Fall back to '$PROCSYSFS' and '$SYSFS' ."
  fi

  set_and_verify "$PROCSYSFS/net/ipv4/tcp_mtu_probing" "0"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_slow_start_after_idle" "0"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_rmem" "4096	131072	6291456"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_wmem" "4096	16384	4194304"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_no_metrics_save" "0"
  set_if_lt "$PROCSYSFS/net/core/somaxconn" "4096"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_max_syn_backlog" "4096"

  # Re-enable default Hystart: HYSTART_ACK_TRAIN (0x1) | HYSTART_DELAY (0x2):
  set_and_verify "$SYSFS/module/tcp_cubic/parameters/hystart_detect" "3"

  if [[ "${GLOBAL_NETBASE_ERROR_COUNTER}" -ne 0 ]]; then
    echo "Setup incomplete and incorrect! Number of Errors: ${GLOBAL_NETBASE_ERROR_COUNTER}"
    exit "${GLOBAL_NETBASE_ERROR_COUNTER}"
  fi

  echo "A3 network tuning teardown completed"
}

main