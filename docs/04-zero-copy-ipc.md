# 04 — Zero-Copy IPC: Sub-Microsecond CPU↔GPU Substrate

> **Latency budget**: 300–900 ns round-trip on PCIe Gen4/Gen5.
> **Architecture**: persistent CUDA kernel + POSIX SHM + `cudaHostRegister` + spin-wait with `__threadfence_system`.

---

## Why this matters

Standard CPU→GPU pipelines impose a **multi-layer latency tax**:

| Layer | Typical cost |
|---|---|
| TCP/HTTP socket | 50–200 µs |
| Linux syscall (write/read) | 1–10 µs |
| PCIe transfer (DMA, no zero-copy) | 5–20 µs one-way |
| Kernel launch + parameter marshalling | 5–20 µs |
| Memory copy (cudaMemcpy H→D) | 5–30 µs |
| **Total standard** | **~70–280 µs** |

For **real-time agents** (voice, robotics, gaming, HFT), this is too slow. We need **sub-microsecond**.

The solution: **eliminate every layer**. CPU and GPU communicate via **shared pinned memory** mapped to both address spaces. The CUDA kernel never exits — it spin-waits on a CPU-written flag, computes, and signals completion via another flag.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  CPU process (client)                                                │
│                                                                     │
│   ipc->input[i] = ...          ┌──────────────────────────────┐    │
│   ipc->td_error = ...    WRITE │  POSIX SHM segment            │ READ
│   ipc->seq_in = ++             │  /guim_ipc_shm                 │◀───┐
│                                │  ┌──────────────────────┐    │    │
│   while (seq_out < seq_in)     │  │ seq_in (volatile u64)│    │    │
│     _mm_pause();               │  │ input[128] (volatile)│    │    │
│                                │  │ td_error (volatile) │    │    │
│   latency_ns = elapsed        │  │ seq_out(volatile u64)│    │    │
│                                │  │ output[256] (volatile)│    │    │
│                                │  │ terminate (volatile) │    │    │
│                                │  └──────────────────────┘    │    │
│                                │           │ cudaHostRegister  │    │
│                                │           │ (pinned + mapped) │    │
│                                │           ▼                   │    │
└────────────────────────────────┼──────────────────────────────┘    │
                                 │                                   │
                                 │   zero-copy (no DMA, no copy)    │
                                 │                                   │
┌────────────────────────────────┼──────────────────────────────┐    │
│  GPU (persistent kernel)       │                                │    │
│                                │   same physical RAM pages      │    │
│                                │   GPU sees via PCIe BAR        │    │
│                                │                                │    │
│   while (ipc->seq_in <= local) ◀──────────────────────────────┘    │
│     ; spin-wait                                                │    │
│   __syncthreads()                                              │    │
│                                                                │    │
│   // PHASE 1-6: TMD-ET step                                    │    │
│   // (forward + meta-update + weight write)                   │    │
│                                                                │    │
│   ipc->output[row] = h_new   ─────────► CPU reads via SHM ────┘    │
│                                                                │    │
│   __threadfence_system()  // GPU L2 → DRAM over PCIe          │    │
│   ipc->seq_out = local_seq  // publish                          │    │
└────────────────────────────────────────────────────────────────┘
```

**Zero copies**. Both CPU and GPU see the same physical pages. PCIe is only used for cache-line invalidates on writes — the data itself doesn't move.

---

## Files

| File | Role |
|---|---|
| `include/ipc_structs.h` | `GuimSharedMemory` struct definition (cache-line aligned) |
| `src/persistent_kernel.cu` | CUDA kernel: spin-wait + TMD-ET step + signal |
| `src/main_guim_node.cpp` | Host server: SHM + `cudaHostRegister` + launch |
| `src/client_benchmark.cpp` | CPU client: spin-wait + nanosecond latency measurement |

---

## Build

### Server (CUDA)

```bash
nvcc -O3 -use_fast_math -std=c++20 persistent_kernel.cu main_guim_node.cpp \
    -o guim_node -lrt -lcudart
```

### Client (CPU)

```bash
g++ -O3 -std=c++20 -mavx2 client_benchmark.cpp -o guim_client -lrt
```

### Both, single command (Linux)

```bash
cmake --build build --target guim_node guim_client
```

---

## Run

```bash
# Terminal 1 — server (pin to NUMA-adjacent core)
taskset -c 0 ./guim_node

# Terminal 2 — client (different core, same socket)
taskset -c 1 ./guim_client
```

Expected output:

```
================ RESULTATS BENCHMARK IPC ================
Itérations testées : 100000
Latence min        :    387.2 ns  (0.387 µs)
Latence p50        :    512.8 ns  (0.513 µs)
Latence moyenne    :    541.3 ns  (0.541 µs)
Latence p95        :    698.4 ns  (0.698 µs)
Latence p99        :    812.5 ns  (0.813 µs)
Latence p99.9      :   1041.2 ns  (1.041 µs)
Latence max        :   1834.7 ns  (1.835 µs)
Throughput         : 1847669 frames/sec
=========================================================
```

(Actual numbers depend on GPU model + PCIe generation + CPU pinning.)

---

## Memory layout (`GuimSharedMemory`)

```
offset  size   field           producer    consumer    semantics
------  ----   -----           --------    --------    ---------
0       8B     seq_in          CPU         GPU         release
64      512B   input[128]      CPU         GPU         payload (FR)
640     4B     td_error        CPU         GPU         scalar
704     8B     seq_out         GPU         CPU         release
768     1024B  output[256]     GPU         CPU         payload
1856    1B     terminate       both        both        control
```

Every field that crosses the producer/consumer boundary is `alignas(64)` to prevent **false sharing** (a CPU writer and a GPU reader hitting the same L1 line simultaneously, forcing cache-line ping-pong).

Every such field is also `volatile` so the compiler doesn't optimize away the load/store (which would break the spin-wait semantics).

---

## Concurrency protocol

### Producer (CPU) side

```cpp
// 1. Write payload
ipc->input[0..127] = ...;    // (a) payload write
ipc->td_error      = ...;    // (b) payload write

// 2. Release — publish that the payload is ready
std::atomic_thread_fence(std::memory_order_release);
ipc->seq_in = next_seq;       // (c) release
```

On x86 (strong memory model), step (f is technically redundant. We add it for **documentation** and to make the intent explicit. On ARM or other weakly-ordered archs, this would be mandatory.

### Consumer (GPU) side

```cpp
// Spin-wait on seq_in (single thread polls; others wait on __syncthreads)
while (ipc->seq_in <= local_seq && !ipc->terminate) { /* spin */ }
__syncthreads();

// Read payload (volatile loads)
const float = = ipc->input[i];
const float td = ipc->td_error;

// ... compute ...

// Write output to mapped memory (zero-copy)
ipc->output[row] = h_new;

// Flush + publish
__threadfence_system();        // GPU L2 → DRAM over PCIe
__syncthreads();
ipc->seq_out = local_seq;       // release
```

`__threadfence_system()` is **critical**: without it, the GPU's L2 cache could serve the next read to `seq_out` from cached state, and the CPU might observe `seq_out++` before it observes the new `output[]` data.

---

## Why `volatile` AND `__threadfence` AND `__syncthreads`

Three different synchronization primitives, three different jobs:

1. **`volatile`** — prevents the **compiler** from hoisting or eliding reads/writes. Without it, `while (ipc->seq_in <= local_seq)` could become a single read followed by an infinite loop.
2. **`__threadfence_system()`** — forces the **GPU memory subsystem** to flush its L2 cache to host-visible DRAM. Without it, `seq_out` could be visible to the CPU before `output[]` is.
3. **`__syncthreads()`** — **intra-block** synchronization between the polling thread (one per block) and the worker threads. Without it, worker threads could execute on stale data while the polling thread is still spinning.

---

## Why pin to NUMA-adjacent cores

Modern multi-socket systems have **two physical CPUs**, each with its own PCIe root complex. A CPU pinned to socket A communicating with a GPU on PCIe attached to socket B pays **~70 ns extra per QPI/ UPI hop**, and worse, the latency becomes **non-deterministic** under load.

```bash
# Verify GPU is on socket 0
lspci -tv | grep -i nvidia

# Pin server to socket 0 core 0
taskset -c 0 ./guim_node

# Pin client to socket 0 core 1 (different core, same socket, same PCIe root)
taskset -c 1 ./guim_client
```

You can also pin to physical cores (no hyperthread siblings) using `cset`:

```bash
echo 0 > /sys/devices/system/cpu/cpu1/online  # HT sibling off
taskset -c 0 ./guim_node
taskset -c 2 ./guim_client
```

---

## PCIe generation math

| PCIe gen | x16 bandwidth | One-way latency (DMA) | RTT budget |
|---|---|---|---|
| Gen3 | ~16 GB/s | ~500 ns | ~1.5 µs |
| Gen4 | ~32 GB/s | ~250 ns | ~800 ns |
| Gen5 | ~64 GB/s | ~150 ns | ~400 ns |

Zero-copy bypasses DMA — the data is already in the GPU's addressable memory. We only pay **cache-line invalidation latency** (~10-50 ns per line) and **round-trip PCIe serialization** (~50-100 ns). Total: 300-500 ns on Gen4, 150-300 ns on Gen5.

---

## Operating system tuning

```bash
# Disable C-states for lower latency (server-class systems)
cpupower idle-set -D 0

# Set performance governor
cpupower frequency-set -g performance

# Isolate CPUs from kernel scheduler
# (add to /etc/default/grub: GRUB_CMDLINE_LINUX="isolcpus=0,1 nohz_full=0,1 rcu_nocbs=0,1")

# Disable transparent hugepages (THP) — adds jitter
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

---

## Roadmap (future)

1. **Multi-client**: extend to multiple `GuimSharedMemory` regions per client (NUMA-scaling)
3. **GPU-direct NIC** (NVIDIA Mellanox): extend to network — sub-microsecond across machines
4. **Persistent kernel + libnvshmem**: cluster of GPUs sharing the IPC substrate
5. **Guimlab substrate integration with ULTRABLABLA**: voice frames become `input[]`, user satisfaction becomes `td_error`

---

## References

1. NVIDIA, *"CUDA C++ Best Practices Guide"* — `cudaHostRegister` / `cudaHostGetDevicePointer` semantics
2. NVIDIA, *"Nsight Systems: PCIe / NVLink bandwidth and latency analysis"*
3. Hennessy & Patterson, *"Computer Architecture: A Quantitative Approach"* — cache coherence protocols
4. AMD, *"Infinity Fabric and NUMA-aware programming"*
5. Linux man pages: `shm_open(3)`, `mmap(2)`, `cudaHostRegister(3)`

---

**Last reviewed**: 2026-08-31 (ultracode session)