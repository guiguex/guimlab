# ==============================================================================
#  Dockerfile — Reproducible CUDA build environment for Guimlab Substrate
#  ==============================================================================
#  Based on nvidia/cuda:12.6.0-devel-ubuntu22.04 — provides nvcc 12.6 + GCC 11.
#
#  Build:
#    docker build -t guimlab:latest .
#
#  Run bench (requires --gpus all and a real GPU):
#    docker run --rm --gpus all --ipc=host guimlab:latest ./build/bin/guim_bench
#
#  Compile-only (no GPU required, for CI):
#    docker build --target builder -t guimlab:build .
# ==============================================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Toolchain + Python (Python is just for benchmark plotting scripts)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build \
        git \
        ca-certificates \
        python3 \
        python3-pip \
        libvulkan-dev \
    && pip install cmake \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Working directory
RUN mkdir -p /guimlab
WORKDIR /guimlab

# Copy source
COPY . .

# Build with CMake
RUN cmake -B build -S . -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
        -DGUIM_BUILD_TESTS=ON \
        -DGUIM_BUILD_BENCH=OFF \
    && cmake --build build --parallel

# ==============================================================================
#  Runtime image — minimal, just the binaries + nvidia runtime libs
# ==============================================================================

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runtime

# Copy built binaries from builder
COPY --from=builder /guimlab/build/bin /guimlab/build/bin
COPY --from=builder /guimlab/scripts /guimlab/scripts

WORKDIR /guimlab/build/bin

# Default entrypoint — drop to bash
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["exec bash"]

# Labels
LABEL org.opencontainers.image.title="GuimLab Substrate"
LABEL org.opencontainers.image.description="Sub-millisecond continual learning neuromorphic substrate in C++/CUDA"
LABEL org.opencontainers.image.source="https://github.com/guiguex/guimlab"
LABEL org.opencontainers.image.vendor="Guillaume Meingan"