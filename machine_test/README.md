# GPUDirect-TCPX Guest OS Machine Tests

## Prerequisite

https://source.corp.google.com/piper///depot/google3/experimental/users/almasrymina/cos-dev/

## Run Stable Tests

### Copy run scripts

```
# on your cloudtop/desktop
./scripts/copy-test-cos <vm1_name> <vm2_name> ...
```

### Run single machine ioctl tests

```
# on your vm
sudo -s
USER=test ./run_tx_ioctl_test # running tx ioctl tests
USER=test ./run_rx_ioctl_test # running rx ioctl tests
```

Example output:

tx: https://paste.googleplex.com/6418888043528192
rx: https://paste.googleplex.com/5023497104392192


### Run two-node single-connection tests

One needs two VMs to run the two-node single-connection tests.  The tests run
the same binary with variant "event handlers" that cover different usage of the
guest OS kernel APIs.

#### Regular TCP tests

##### Basic TCP send/receive Test

Server:

```
# on the server
sudo -s
export SERVER=<server ip> CLIENT=<client ip>
USER=test ./run_connection_test --server_address $SERVER --server --event_handler tcp_send

# Note: wait until you see "[Server] Starting Server Control Channel ..." trace
# before starting the client
```

Client:

```
# on the client
sudo -s
export SERVER=<server ip> CLIENT=<client ip>
USER=test ./run_connection_test --server_address $SERVER --event_handler tcp_receive

```

Example output:

server[tcp-send]: https://paste.googleplex.com/5645195688280064
client[tcp-receive]: https://paste.googleplex.com/6238653813620736

##### More TCP variants

See
[src](https://source.corp.google.com/h/team/kernel-net-team/tcpgpudmarxd/+/dev:machine_test/cuda/event_handler_factory.cu;l=158)
for the list of event handlers.  Replace the `--event_handler` argument with
the desired event handlers when running the binary as described above.

Example output:

server[tcp-send-tcp-direct]: https://paste.googleplex.com/4543385481248768
client[tcp-receive]: https://paste.googleplex.com/5072577574731776

server[tcp-send]: https://paste.googleplex.com/5691089641013248
client[tcp-receive-tcp-direct]: https://paste.googleplex.com/5422190328545280

#### TCPX GPU TCP tests

#### Basic GPU send/receive Test

Server:

```
# on the server
sudo -s
export SERVER=<server ip> CLIENT=<client ip>
USER=test IS_SERVER ./start_tcpxd_cos
# Note: wait until you see "Rx Rule Manager server(s) started" before starting the test
USER=test ./run_connection_test --server_address $SERVER --server --event_handler gpu_send
# Note: wait until you see "[Server] Starting Server Control Channel ..." trace
# before starting the client
```

Client:

```
# on the client
sudo -s
export SERVER=<server ip> CLIENT=<client ip>
USER=test ./start_tcpxd_cos
# Note: wait until you see "Rx Rule Manager server(s) started" before starting the test
USER=test ./run_connection_test --server_address $SERVER --event_handler gpu_receive
```

Example output:

server[gpu-send]: https://paste.googleplex.com/5302189747601408
client[gpu-receive]: https://paste.googleplex.com/4876266430791680

#### 1B to 1GB message size GPU send/receive Test

Server:

```
# on the server
sudo -s
export SERVER=<server ip> CLIENT=<client ip>
USER=test IS_SERVER ./start_tcpxd_cos
# Note: wait until you see "Rx Rule Manager server(s) started" before starting the test
USER=test ./run_message_size_test --server_address $SERVER --server --event_handler gpu_send
# Note: wait until you see "[Server] Starting Server Control Channel ..." trace
# before starting the client
```

Client:

```
# on the client
sudo -s
export SERVER=<server ip> CLIENT=<client ip>
USER=test ./start_tcpxd_cos
# Note: wait until you see "Rx Rule Manager server(s) started" before starting the test
USER=test ./run_message_size_test --server_address $SERVER --event_handler gpu_receive
```

##### More TCPX tests variants

Note: Running the TCPX GPU tests, one needs the TCPX Rx manager running in the
background, which is done in the "start_tcpxd_cos" script run in the commands above.

See
[src](https://source.corp.google.com/h/team/kernel-net-team/tcpgpudmarxd/+/dev:machine_test/cuda/event_handler_factory.cu;l=158)
for the list of event handlers.  Replace the `--event_handler` argument with
the desired event handlers when running the binary as described above.

Example output:

sender[variants]: https://paste.googleplex.com/4770894965637120
receiver[variants]: https://paste.googleplex.com/5165530464911360

## Build and push custom tests

### Build and push custom images

```
cd ../ # go to tcpgpudmarxd root src folder
docker build --network=host -t gcr.io/a3-tcpd-staging-hostpool/$USER/machine_test -f Dockerfile.machine_test .
docker push gcr.io/a3-tcpd-staging-hostpool/$USER/machine_test
```

### Run the Tests

Follow the steps in [Run Stable Tests](#run-stable-tests), replace "stable" with your $USER.
