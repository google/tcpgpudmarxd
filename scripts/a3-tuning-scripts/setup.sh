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

# Please use verbose as the first param
if [[ $* == *"--verbose"* ]]; then
  shift
  set -x
fi

set_route_param() {
  initcwnd=$1
  rto_min=$2
  quickack=$3
  suffix=

  # Note that suffix strings are added in the reverse of the
  # order in which they are removed:

  if [ "$#" -eq 0 ]; then
      echo "No routes are modified"
      return
  fi

  if [[ ! "${initcwnd}" -eq 0 ]]; then
    suffix="initcwnd ${initcwnd}"
  fi

  if [[ ! "${rto_min}" -eq 0 ]]; then
    suffix="${suffix} rto_min ${rto_min}ms"
  fi

  if [[ ! "${quickack}" -eq 0 ]]; then
    suffix="${suffix} quickack 1"
  fi

  if [[ -z "$suffix" ]]; then
    return
  fi

  while read -r line; do
    ip route del ${line} 2> /dev/null
    ip route add ${line} ${suffix} 2> /dev/null
  done < <(ip route show)

  ip route show
}

main() {
  set_route_param $1 $2 $3

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
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_rmem" "4096	1048576	15728640"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_wmem" "4096	1048576	67108864"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_no_metrics_save" "1"
  set_if_lt "$PROCSYSFS/net/core/somaxconn" "4096"
  set_and_verify "$PROCSYSFS/net/ipv4/tcp_max_syn_backlog" "4096"

  # For TCP CUBIC Hystart just use: HYSTART_DELAY (0x2).
  # The HYSTART_ACK_TRAIN (0x1) mechanism has signiificant false positive risk;
  # particularly when pacing is enabled, but potentially in other cases, too.
  set_and_verify "$SYSFS/module/tcp_cubic/parameters/hystart_detect" "2"

  if [[ "${GLOBAL_NETBASE_ERROR_COUNTER}" -ne 0 ]]; then
    echo "Setup incomplete and incorrect! Number of Errors: ${GLOBAL_NETBASE_ERROR_COUNTER}"
    exit "${GLOBAL_NETBASE_ERROR_COUNTER}"
  fi

  echo "A3 network tuning v1.0.5 setup completed"
}

main $@
