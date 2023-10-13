#!/bin/bash

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
reset_route_param
echo 0 > /proc/sys/net/ipv4/tcp_no_metrics_save
echo "4096 131072 6291456" > /proc/sys/net/ipv4/tcp_rmem
echo "4096 16384 4194304" > /proc/sys/net/ipv4/tcp_wmem
echo 20480 > /proc/sys/net/core/optmem_max

# Re-enable default Hystart: HYSTART_ACK_TRAIN (0x1) | HYSTART_DELAY (0x2):
echo 3 > /sys/module/tcp_cubic/parameters/hystart_detect

echo "A3 network tuning teardown completed"
