(http://go/tcpx-rxdm)

The Receive Data Path Manager enables zero mem-copy on the NIC-to-GPU
data path for the ingress data stream of the cross-host GPU-heavy
workloads using the GPUDirect-TCPX (formerly TCPDirect) feature.

## Overview

- Design: http://go/tcpx-rx-manager
- Team Review: http://go/tcpdrxm-team-review

## Team

chechenglin@, wwchao@

## Buganizer Hotlist

- A3 (Track 1): https://b.corp.google.com/hotlists/4890613

## Development Guide

- [Build and Run on A3 COS VM](https://source.corp.google.com/h/team/kernel-net-team/tcpgpudmarxd/+/dev:docs/DEV-GUIDE.md)
- [Machine-to-Machine Test on A3 COS VM](https://source.corp.google.com/h/team/kernel-net-team/tcpgpudmarxd/+/dev:machine_test/README.md)

## Release

- Release Notes: http://go/tcpx-rxdm-release-notes
- Release Process: http://go/tcpx-rxdm-release-process

## See Also

- http://go/tcpdirect-architecture
- http://go/tcpdirect-rx-buf-design
