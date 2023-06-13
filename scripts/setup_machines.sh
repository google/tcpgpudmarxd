#!/bin/sh
#

set -ex


while true; do
  machine=$1
  shift || break

  # mkdirs
  ssh root@$machine "mkdir -p /export/hda3/$USER/tcpgpudmad"
  ssh root@$machine "mkdir -p /export/hda3/$USER/tcpdirect-bench"

  # Copy validation scripts
  scp scripts/pre_tcpgpudmad.sh root@$machine:/export/hda3/$USER/tcpgpudmad/

  # Copy validation binaries
  pushd ../../../../blaze-bin/experimental/users/chechenglin/tcpgpudmad/
    scp app/tcpgpudmad \
      benchmark/test/tcpdirect_single_connection_test \
      benchmark/test/tcpdirect_benchmark \
      benchmark/test/tcpdirect_invalid_ioctl_test \
      root@$machine:/export/hda3/$USER/tcpgpudmad/
  popd

  # Copy tcpdirect-bench scripts
  pushd ../../kaiyuanz/tcpdirect-bench/
    rsync -a --progress \
      *.sh \
      root@$machine:/export/hda3/$USER/tcpdirect-bench/
  popd

  # Copy tcpdirect-bench binaries
  pushd ../../../../blaze-bin/experimental/users/kaiyuanz/tcpdirect-bench/

    rsync -a --progress  nvdmad_multigpu_dmabuf \
      root@$machine:/export/hda3/$USER/tcpdirect-bench/

    rsync -a --progress multinic_tcpdirect_bench \
      root@$machine:/export/hda3/$USER/tcpdirect-bench/

  popd

  echo
  echo Setup machine $machine complete...
  echo

done

echo Setup machines complete
