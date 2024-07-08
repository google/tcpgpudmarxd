## Overview

The Receive Data Path Manager (RxDM) enables zero mem-copy on the NIC-to-GPU
data path for the ingress data stream of the cross-host GPU-heavy
workloads using the GPUDirect-TCPX (formerly TCPDirect) feature. The public guide for using GPUDirect-TCPX project is https://cloud.google.com/compute/docs/gpus/gpudirect.


## Release

The latest version of RxDM can be found in [Artifact Registry](us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/tcpgpudmarxd-dev). We recommend to use version tag (e.g. v2.0.12) instead of using latest tag to specify the version for better debugging.
