FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND='noninteractive'

RUN apt update \
  && apt-get install -y --no-install-recommends \
        git openssh-server wget iproute2 vim build-essential cmake gdb \
	protobuf-compiler libprotobuf-dev rsync ethtool \
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

# copy all license files
WORKDIR /third_party/licenses
RUN cp ../abseil-cpp/LICENSE license_absl.txt

COPY . /tcpgpudmarxd

WORKDIR /tcpgpudmarxd
RUN rm -rf build docker-build
WORKDIR build
RUN cmake -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release ..
RUN make -j
WORKDIR test
RUN ctest

WORKDIR /tcpgpudmarxd
USER root
ENTRYPOINT /tcpgpudmarxd/build/tcpgpudmarxd
