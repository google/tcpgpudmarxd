## Build the image and push it to your project repository

```
$ docker build -t <GCR path> -f Dockerfile .
$ docker push <GCR path>
```

If you see DNS errors as the ones below when building the docker image,

```
Ign:1 http://archive.ubuntu.com/ubuntu jammy InRelease
Ign:2 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
Ign:2 http://security.ubuntu.com/ubuntu jammy-security InRelease
0% [Connecting to archive.ubuntu.com] [Connecting to security.ubuntu.com] [ConnectinIgn:2 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:5 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Ign:1 http://archive.ubuntu.com/ubuntu jammy InRelease
Err:2 http://security.ubuntu.com/ubuntu jammy-security InRelease
  Temporary failure resolving 'security.ubuntu.com'
Err:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
  Temporary failure resolving 'developer.download.nvidia.com'
0% [Connecting to archive.ubuntu.com]
```

Adding `--network=host` to the build command may resolve the issue.

```
$ docker build --network=host -t <GCR path> -f Dockerfile .
```

If you see the following errors when trying to push the docker image:

```
unauthorized: You don't have the needed permissions to perform this operation,
and you may have invalid credentials. To authenticate your request, follow the steps
in: https://cloud.google.com/container-registry/docs/advanced-authentication

```

You can try:

```
gcloud docker -- push <GCR path>
```

or for staging:

```
staging_gcloud docker -- push <GCR path>
```

Re-installing docker also helps.  Instructions are available in:
https://g3doc.corp.google.com/cloud/containers/g3doc/glinux-docker/install.md?cl=head


## Launch the image

Use the following script to launch the container on VMs:

```
#!/bin/bash

function run_tcpgpudmarxd() {
  docker run --pull=always -it --rm \
    --cap-add=NET_ADMIN --network=host \
    --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidia1:/dev/nvidia1 \
    --device /dev/nvidia2:/dev/nvidia2 \
    --device /dev/nvidia3:/dev/nvidia3 \
    --device /dev/nvidia4:/dev/nvidia4 \
    --device /dev/nvidia5:/dev/nvidia5 \
    --device /dev/nvidia6:/dev/nvidia6 \
    --device /dev/nvidia7:/dev/nvidia7 \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    --volume /tmp:/tmp \
    --entrypoint /tcpgpudmarxd/build/app/tcpgpudmarxd \
    <GCR path> "$@"
}
```

The source code could be found in `/tcpgpudmarxd`, while the built binary could be found in `/tcpgpudmarxd/build`

