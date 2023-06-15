## Build the image and push it to your project repository

```
$ docker build -t <GCR path> -f Dockerfile .
$ docker push <GCR path>
```

After that, use the following script to launch the container on VMs:

```
#!/bin/bash

function run_tcpgpudmarxd() {
  docker run --pull=always -it --rm \
    -u 0 --cap-add=NET_ADMIN --network=host \
    --pid=host \
    --userns=host \
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
  	<GCR path> "$@"
}
```

The source code could be found in `/tcpgpudmarxd`, while the built binary could be found in `/tcpgpudmarxd/build`

