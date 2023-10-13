#!/bin/bash

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

set_route_param $1 $2 $3
echo 0 > /proc/sys/net/ipv4/tcp_mtu_probing
echo 0 > /proc/sys/net/ipv4/tcp_slow_start_after_idle
echo 1 > /proc/sys/net/ipv4/tcp_no_metrics_save
echo "4096 1048576 15728640" > /proc/sys/net/ipv4/tcp_rmem
echo "4096 1048576 67108864" > /proc/sys/net/ipv4/tcp_wmem
echo 131072 > /proc/sys/net/core/optmem_max
echo 4096 > /proc/sys/net/core/somaxconn
echo 4096 > /proc/sys/net/ipv4/tcp_max_syn_backlog

# For TCP CUBIC Hystart just use: HYSTART_DELAY (0x2).
# The HYSTART_ACK_TRAIN (0x1) mechanism has signiificant false positive risk;
# particularly when pacing is enabled, but potentially in other cases, too.
echo 2 > /sys/module/tcp_cubic/parameters/hystart_detect

echo "A3 network tuning v1.0.4 setup completed"
