# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND='noninteractive'

RUN apt update \
  && apt-get install -y --no-install-recommends \
        git openssh-server wget iproute2 vim build-essential cmake gdb \
        protobuf-compiler libprotobuf-dev libprotoc-dev rsync libssl-dev \
        pkg-config libmnl-dev \
  && rm -rf /var/lib/apt/lists/*

# build absl
WORKDIR /third_party
RUN git clone https://github.com/abseil/abseil-cpp.git
WORKDIR abseil-cpp
RUN git fetch --all --tags
RUN git checkout tags/20230125.3 -b build
WORKDIR build
RUN cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DABSL_USE_GOOGLETEST_HEAD=ON ..
RUN cmake --build . -j 8 --target all && cmake --install .

# build gRPC
WORKDIR /third_party
RUN git clone -b v1.32.0 https://github.com/grpc/grpc
WORKDIR grpc
RUN git submodule update --init
WORKDIR build
RUN cmake \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DgRPC_ABSL_PROVIDER=package \
    -DgRPC_SSL_PROVIDER=package \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DgRPC_INSTALL=ON .. \
    && make -j && make install

# build ethtool
WORKDIR /third_party
RUN wget https://mirrors.edge.kernel.org/pub/software/network/ethtool/ethtool-6.3.tar.gz
RUN tar -xvf ethtool-6.3.tar.gz
WORKDIR ethtool-6.3
RUN ./configure && make && make install

# copy all license files
WORKDIR /third_party/licenses
RUN cp ../abseil-cpp/LICENSE license_absl.txt
RUN cp ../ethtool-6.3/LICENSE license_ethtool.txt

COPY . /tcpgpudmarxd

ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib"
WORKDIR /tcpgpudmarxd
RUN rm -rf build docker-build
WORKDIR build
RUN cmake -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release ..
RUN make -j && make install
RUN cp -r ../scripts/a3-tuning-scripts .
WORKDIR test
RUN ctest

WORKDIR /tcpgpudmarxd
RUN ls | grep -v "build\|LICENSE" | xargs rm -rf
USER root
ENTRYPOINT /tcpgpudmarxd/build/app/tcpgpudmarxd
