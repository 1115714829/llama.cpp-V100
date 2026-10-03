# V100 (sm_70) image of llama.cpp-v100, following the structure of .devops/cuda.Dockerfile.
#
# One Dockerfile for both platforms, built per platform and combined into one multi-arch tag:
#   linux/amd64    any x86_64 host with V100 GPUs
#   linux/ppc64le  IBM Power AC922 only (POWER9 + V100-SXM2 + NVLink); other ppc64le hosts are not supported
#
# The base is the CUDA 12.4 UBI8 image because it is the CUDA image published for both amd64 and
# ppc64le; NCCL (required for multi-GPU tensor split) comes with it. Only sm_70 SASS is built.
#
#   docker build -f .devops/v100-cuda.Dockerfile -t llama.cpp-v100:server .
#
# The host needs an NVIDIA driver that supports CUDA 12.4 (550 or newer) and a container runtime
# that can pass the GPUs into the container (NVIDIA Container Toolkit, or a CDI spec).

ARG CUDA_VERSION=12.4.1
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubi8
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubi8

# ---------- build ----------
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

RUN dnf install -y gcc-toolset-12-gcc-c++ cmake git-core && dnf clean all

WORKDIR /src
COPY . .

# NCCL is mandatory: without it multi-GPU tensor split has no collective, so the build fails instead
RUN source /opt/rh/gcc-toolset-12/enable && \
    test -f /usr/include/nccl.h && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=70-real \
        -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_NCCL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON 2>&1 | tee cmake.log && \
    ! grep -q "NCCL not found" cmake.log && \
    cmake --build build -j "$(nproc)" --target llama-server llama-cli llama-quantize

RUN mkdir -p /app/licenses && \
    cp -a build/bin/llama-server build/bin/llama-cli build/bin/llama-quantize build/bin/lib*.so* /app/ && \
    cp -a /usr/lib64/libnccl.so* /app/ && \
    cp LICENSE README.md /app/ && \
    cp licenses/* ggml/src/ggml-cuda/LICENSE.v100-skinny ggml/src/ggml-cuda/gdn-chunk-sm70/LICENSE-tilelang \
       ggml/src/ggml-cuda/sm70-vendor/LICENSE-* /app/licenses/

# ---------- ffmpeg (video input), from conda-forge: available for both linux-64 and linux-ppc64le ----------
FROM ${BASE_CUDA_DEV_CONTAINER} AS ffmpeg

RUN dnf install -y bzip2 && dnf clean all && \
    case "$(uname -m)" in \
        x86_64)  arch=64 ;; \
        ppc64le) arch=ppc64le ;; \
        *) echo "unsupported architecture $(uname -m)"; exit 1 ;; \
    esac && \
    curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-${arch}/latest" | tar -xj -C /usr/local bin/micromamba && \
    micromamba create -y -p /opt/ffmpeg -c conda-forge ffmpeg && \
    micromamba clean -a -y

# ---------- server (default) ----------
FROM ${BASE_CUDA_RUN_CONTAINER} AS server

RUN dnf install -y libgomp && dnf clean all

COPY --from=build /app /app
COPY --from=ffmpeg /opt/ffmpeg /opt/ffmpeg

ENV PATH=/opt/ffmpeg/bin:/app:${PATH} \
    LLAMA_ARG_HOST=0.0.0.0

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
