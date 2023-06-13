# Quick Start (building, running the tests, and running the benchmark)

## Build

```
blaze build -c opt --config=cuda12 //experimental/users/chechenglin/tcpgpudmad/...
blaze build -c opt --config=cuda12 //experimental/users/kaiyuanz/tcpdirect-bench/...
```

## Setup Machine

Currently it supports Adastra PreDvt (mvff1, mvff2, mvii5, mvii6)

Using mvff1, mvff2 pair as an example here.

### Copy scripts and binaries

```
./scripts/setup_machines.sh mvff1 mvff2
```

### Run Scripts

On both mvff1,mvff2 `cd /export/hda3/$USER/tcpgpudmad sh pre_tcpgpudmad.sh`

## Run Tests

Note: to run loopback test, simply running the server and client on the same
host.

### simple tcp_send + tcp_receive

On mvff1

```
$ cd /export/hda3/$USER
$ container.py run --overwrite \
--memlimit=unlimited --network-max=1000000 --ram=4096 test -- \
./tcpdirect_single_connection_test --alsologtostderr --server_address \
2002:af4:3480:1241:: --port 50001 --server --event_handler tcp_send \
--message_size 1638400
```

On mvff2

```
$ cd /export/hda3/$USER
$ container.py run --overwrite \
--memlimit=unlimited --network-max=1000000 --ram=4096 test -- \
./tcpdirect_single_connection_test --alsologtostderr --server_address \
2002:af4:3480:1241:: --port 50001 --event_handler tcp_receive \
--message_size 1638400
```

### simple gpu_send + gpu_receive

On mvff1

```
$ cd /export/hda3/$USER
$ ./tcpgpudmad &
$ container.py run --overwrite \
--memlimit=unlimited --network-max=1000000 --ram=4096 test -- \
./tcpdirect_single_connection_test --alsologtostderr --server_address \
2002:af4:3480:1241:: --port 50001 --server --event_handler gpu_send \
--message_size 1638400
```

On mvff2

```
$ cd /export/hda3/$USER
$ ./tcpgpudmad &
$ container.py run --overwrite \
--memlimit=unlimited --network-max=1000000 --ram=4096 test -- \
./tcpdirect_single_connection_test --alsologtostderr --server_address \
2002:af4:3480:1241:: --port 50001 --event_handler gpu_receive \
--message_size 1638400
```
### Event Handlers

As you may have noted from the commands above, we can switch out the event
handlers for different senders/receivers. Here is a list of availalbe
As you may have noted from the commands above, we can switch out the event handlers for different senders/receivers.  Here is a list of availalbe

#### GPU Receivers
- gpu_receive
- gpu_receive_dummy_pci
- gpu_receive_miss_flag
- gpu_receive_mix_tcp
- gpu_receive_token_free
- gpu_receive_no_token_free

#### GPU Senders
- gpu_send
- gpu_sender_dummy_page
- gpu_send_miss_flag
- gpu_send_mix_tcp
- gpu_send_oob

#### TCP Receivers
- tcp_receive
- tcp_receive_tcp_direct

#### TCP Senders
- tcp_send
- tcp_send_tcp_direct

## Run Benchmark

### Run Scripts

On both mvff1,mvff2:

```
cd /export/hda3/$USER/tcpdirect-bench
source setup_nvdma_dmabuf_predvt.sh
```

### Run benchmark

On mvff1

```
 ./tcpdirect_benchmark --alsologtostderr --server_address 2002:af4:3480:1241:: \
 --server --port 50001 --num_gpus 4 --socket_per_thread 2 --threads_per_gpu \
 8 --message_size 409600 --use_dmabuf
```

On mvff2

```
 ./tcpdirect_benchmark --alsologtostderr --server_address 2002:af4:3480:1241:: \
 --port 50001 --num_gpus 4 --socket_per_thread 2 --threads_per_gpu 8 \
 --message_size 409600 --use_dmabuf
```

### Result

On mvff1

```
```

On mvff2

```
```

### Manually defined GPU-NIC topology

Textproto can be used to specify GPU-NIC topology. Users can either use preset
GPU-NIC pairs topology or use manual option. Manual option allows them to pass
the path of the textproto file using ```gpu_nic_topology``` flag to use the
self-defined GPU-NIC toppology.

For example, the command to use gpu_nic_mismatch.textproto to test mismatching
tx nic and socket: server: ```container.py run --overwrite --memlimit=unlimited
--network-max=1000000 --ram=4096 test --
./benchmark/test/tcpdirect_single_connection_test --alsologtostderr
--server_address 2002:aa2:c641:: --port 50001 --server --event_handler gpu_send
--message_size 1638400 --gpu_nic_preset monstertruck```

client: ```container.py run --overwrite --memlimit=unlimited --network-max=1000000
--ram=4096 test -- ./benchmark/test/tcpdirect_single_connection_test
--alsologtostderr --server_address 2002:aa2:c641:: --port 50001 --event_handler
gpu_receive --message_size 1638400 --gpu_nic_index 0 --gpu_nic_preset
manual --gpu_nic_topology gpu_nic_mismatch.textproto```
