#!/bin/bash

source "/tcpgpudmarxd/build/a3-tuning-scripts/sysctl_helper.sh"
GLOBAL_NETBASE_ERROR_COUNTER=0

# Please use verbose as the first param
if [[ $* == *"--verbose"* ]]; then
  shift
  set -x
fi


main() {
  python3 /tcpgpudmarxd/build/a3-tuning-scripts/tuning_persistence.py $1 $2 $3 &
  echo "A3 network tuning persistence thread started"

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

}

main $@
