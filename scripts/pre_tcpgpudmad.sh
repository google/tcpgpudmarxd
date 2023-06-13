#!/bin/bash

# These are scripts for monstertruck machines setting up the system before
# running tcpgpudmad.  The assumption here is: these will be setup by the Guest
# OS.  If the assumption changes we should add the necessary setup into
# tcpgpudmad.

set_rps_inner() {
  local IFNAME=$1;
  local NUM_Q=$2;

  for i in `seq 0 $(($NUM_Q - 1))`; do
    echo 16384 > /sys/class/net/${IFNAME}/queues/rx-$i/rps_flow_cnt;
  done
}

set_rps() {
  local NUM_Q=$1;

  set_rps_inner dcn1 ${NUM_Q};
  set_rps_inner dcn2 ${NUM_Q};
  set_rps_inner dcn3 ${NUM_Q};
  set_rps_inner dcn4 ${NUM_Q};
}

set_rps_cpus_inner() {
  local IFNAME=$1;
  local NUM_Q=$2;
  local IRQ_START=$3;

  for i in `seq 0 $(($NUM_Q - 1))`; do
    local CPUS=`cat /proc/irq/$(($IRQ_START + $i))/smp_affinity`;
    echo ${CPUS} > /sys/class/net/${IFNAME}/queues/rx-$i/rps_cpus;
  done
}

set_rps_cpus() {
  local NUM_Q=$1;
  set_rps_cpus_inner dcn1 ${NUM_Q} 1062;
  set_rps_cpus_inner dcn2 ${NUM_Q} 550;
  set_rps_cpus_inner dcn3 ${NUM_Q} 1576;
  set_rps_cpus_inner dcn4 ${NUM_Q} 2088;
}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/libcuda

/usr/lib/libcuda/cuda_health_check

NUM_Q=14

modprobe nvp2p_dma_buf
modprobe nvp2p_dma_glue

set_rps ${NUM_Q}
set_rps_cpus ${NUM_Q}
