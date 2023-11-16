#!/bin/bash

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

  set_and_verify "/proc/sys/net/ipv4/tcp_mtu_probing" "0"
  set_and_verify "/proc/sys/net/ipv4/tcp_slow_start_after_idle" "0"
  set_and_verify "/proc/sys/net/ipv4/tcp_rmem" "4096	131072	6291456"
  set_and_verify "/proc/sys/net/ipv4/tcp_wmem" "4096	16384	4194304"
  set_and_verify "/proc/sys/net/ipv4/tcp_no_metrics_save" "0"
  set_and_verify "/proc/sys/net/core/optmem_max" "20480"
  set_if_lt "/proc/sys/net/core/somaxconn" "4096"
  set_and_verify "/proc/sys/net/ipv4/tcp_max_syn_backlog" "4096"

  # Re-enable default Hystart: HYSTART_ACK_TRAIN (0x1) | HYSTART_DELAY (0x2):
  set_and_verify "/sys/module/tcp_cubic/parameters/hystart_detect" "3"

  if [[ "${GLOBAL_NETBASE_ERROR_COUNTER}" -ne 0 ]]; then
    echo "Setup incomplete and incorrect! Number of Errors: ${GLOBAL_NETBASE_ERROR_COUNTER}"
    exit "${GLOBAL_NETBASE_ERROR_COUNTER}"
  fi

  echo "A3 network tuning teardown completed"
}

main