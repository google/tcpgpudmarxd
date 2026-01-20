# TCPX RxDM

## Overview

The Receive Data Path Manager (RxDM) enables zero mem-copy on the NIC-to-GPU
data path for the ingress data stream of the cross-host GPU-heavy
workloads using the GPUDirect-TCPX (formerly TCPDirect) feature. The public guide of using GPUDirect-TCPX project is https://cloud.google.com/compute/docs/gpus/gpudirect.

## Getting Started

### Building

To simplify the build process, we have conveniently provided a build script:

- [`rxdm_build.sh`](rxdm_build.sh)

The script prepares the required dependencies then builds a Docker image containing
the Receive Data Path Manager (RxDM). It is highly configurable and contains flags to
specify the repository to which the Docker image should be pushed.

Sample usage:
```
./rxdm_build.sh -p -c -r $SAMPLE_REPO -i $SAMPLE_IMAGE_NAME -t $SAMPLE_TAG
```

## Contributing

Contributions are always welcomed. Please refer to our [contributing guidelines](docs/contributing.md)
to learn how to contriute.

## License

RxDM GPUDirectTCPX is licensed under the terms of a BSD-style license.
See [LICENSE](LICENSE) for more information.
