# CUDA Engineering — Fused Kernels, Arenas, and Continual-Learning Performance

> **Audience** : kernel authors building a low-latency neuromorphic continual-learning runtime where inference, eligibility trace, weight update, and plasticity reset must complete in a single deterministic pass per frame.
> **Target hardware** : NVIDIA Ampere (A100 / RTX 30) minimum, Ada Lovelace and Hopper (H100 / H200) preferred. Blackwell B200 / RTX 50 noted where Blackwell-specific guidance diverges.
> **Latency budget** : < 50 μs per frame for the inner loop (single recurrent layer + plasticity step).
> **Toolchain** : CUDA Toolkit 12.x / 13.x, `nvcc`, CMake ≥ 3.26, Nsight Compute, Nsight Systems.

---

## 1. Fused Kernels — Principles

### 1.1 Why fuse?

GPU kernels are bound by either **arithmetic throughput** (FLOPS) or **memory bandwidth** (bytes/s). A kernel that does not fuse typically:

1. Writes an intermediate tensor to HBM (global memory).
2. Launches a second kernel that re-reads that tensor from HBM.
3. Writes another intermediate, and so on.

Each round-trip costs ~700–900 GB/s of effective bandwidth and a kernel-launch overhead of 3–10 μs. On a continual-learning inner loop of four logical phases (forward → trace → weight update → plasticity reset), the un-fused cost is ~25–40 μs just in launch latency — already over the budget.

Fusing collapses the phases so that intermediates live in **registers** (fastest) or **shared memory** (next-fastest), and only the input/output tensors ever touch HBM. The cost saved per round-trip is:

```
bytes_saved = sum(intermediate_tensor_sizes) * element_size
time_saved ≈ bytes_saved / mem_bandwidth
         + (num_kernels_removed) * launch_overhead
```

For our target:

| Phase          | Intermediate size (FP16, hidden=4096) | Saved HBM traffic | Saved time @ 800 GB/s |
|----------------|----------------------------------------|-------------------|------------------------|
| Forward→Trace  | 4096 × 2 B = 8 KB                      | 8 KB              | ~10 ns                 |
| Trace→Update   | 4096² × 2 B = 32 MB (weight deltas)    | 32 MB             | ~40 μs                 |
| Update→Reset   | 4096 × 4 B = 16 KB (variance)          | 16 KB             | ~20 ns                 |

The dominant saving is the **trace→update** edge, because the eligibility trace must be read+written by every output neuron. Fusing these is non-negotiable for our budget.

Sources:
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
- [Aman's AI Journal — FlashAttention](https://aman.ai/primers/ai/flashattention/)
- [FlashAttention-2 (OpenReview)](https://openreview.net/forum?id=mZn2Xyh9Ec)
- [FlashAttention-4 (arXiv, 2026)](https://arxiv.org/html/2603.05451v1)

### 1.2 Common fusion patterns

| Pattern                                  | Example                                              | Source                              |
|------------------------------------------|------------------------------------------------------|-------------------------------------|
| **GEMM + bias + activation epilogue**    | CUTLASS `epilogue::LinearCombinationRelu`            | NVIDIA CUTLASS                      |
| **Softmax + matmul (online normalizer)** | FlashAttention-1/2/4                                 | Dao et al.                           |
| **RMSNorm + QKV projection**             | LLaMA inference kernels (vLLM, xformers)              | vLLM, xformers                      |
| **Forward + backward in one pass**       | FlashAttention-2 backward; FlashInfer                 | FlashInfer                          |
| **Forward + eligibility trace + update** | This document                                        | Novel                                |
| **KV-cache attention with paging**       | vLLM PagedAttention                                   | Kwon et al., SOSP '23               |

### 1.3 The fundamental tradeoff: register pressure vs occupancy

Every value held across a `__syncthreads()` or between arithmetic phases must live in a register. If a fused kernel needs more live registers than the SM can supply per thread (hardware limit: 255 per thread, 64K per SM, typically capped at 32–64 by `--maxrregcount` to retain occupancy), the compiler **spills** to local memory (cached in L1) — instantly negating the fusion benefit.

The occupancy curve for a fused kernel on Ampere typically looks like:

```
registers/thread   occupancy (%)   notes
≤ 32               100             ideal — 1 fused phase
33–48              75–66           2 fused phases OK
49–64              50              borderline
> 64               < 50            spilling starts; re-design
```

Mitigations:
1. **Reduce live values**: compute the weight update in-place over the trace; don't keep both `trace` and `delta_w` alive simultaneously.
2. **Recompute instead of store**: re-derive activations from a checkpoint rather than holding intermediates.
3. **Lower precision intermediates**: store the trace in FP16 even if forward runs in FP32 (with stochastic rounding on the cast).
4. **Tile the work**: a 4096-wide layer tiled into 4 chunks of 1024 needs 4× less register pressure than the full layer.

### 1.4 Reference implementations

**FlashAttention** is the canonical example of a fused kernel. It computes QK^T, mask, softmax, dropout, and the final V multiply in one kernel by:
- Loading Q, K, V tiles into shared memory
- Computing attention scores in registers
- Maintaining an **online softmax normalizer** (running max + running sum)
- Writing the final output tile to HBM only once

**FlashAttention-2/4** (Dao et al., 2023 / 2026) further parallelizes over the query dimension and pipelines SRAM loads with compute.

**CUTLASS** exposes composable building blocks: `cute::GEMM`, `cute::Epilogue`, `cute::Swizzle`. A custom fused epilogue can be written by subclassing `cutlass::epilogue::thread::LinearCombination` and adding the plasticity reset as a post-matmul step inside the epilogue visitor tree.

---

## 2. Memory Arena — Carmack-style

### 2.1 The Quake / id Tech philosophy

John Carmack's memory model for Quake / Quake III / Doom 3 was a **zone (arena) allocator**:
1. At level start, allocate a large contiguous block once (`Hunk_Alloc`).
2. During the level, bump a pointer for each allocation.
3. At level end, **single** reset — every allocation is freed at once.

The rationale: deterministic, predictable, no fragmentation, no malloc/free on the hot path.

In CUDA, the hot path is the per-frame inner loop. Any `cudaMalloc` / `cudaFree` inside it:
- Takes ~1–10 μs per call (even when served from a pool).
- Holds a global mutex that serializes with all other streams.
- Can trigger `cudaErrorMemoryAllocation` under pressure.

Sources:
- [Quake Arena Allocator (jfo/codearchitect)](https://github.com/jfo/codearchitect/blob/master/notes/Quake%20Arena%20Allocator.md)
- [A Quick Look at the Quake Source Code (fabiensanglard.net)](https://fabiensanglard.net/quake/index.php)
- [Memory Management: A Look at Quake 3 (fabiensanglard.net)](https://fabiensanglard.net/quake3/memory_management.php)

### 2.2 CUDA bump allocator

```cuda
// arena_globals.cuh
struct alignas(128) ArenaState {
    uint8_t* base;            // device pointer to arena base
    size_t   capacity;        // total size in bytes
    size_t   offset;          // current bump offset (atomic for safety)
};
extern __device__ ArenaState g_arena;

__device__ inline void* arena_alloc(size_t n, size_t align = 128) {
    size_t aligned = ((n + align - 1) / align) * align;
    size_t off = atomicAdd((unsigned long long*)&g_arena.offset, aligned);
    return (off + n <= g_arena.capacity) ? (g_arena.base + off) : nullptr;
}

__device__ inline void arena_reset() {
    // single-threaded reset: launch 1 thread, 1 block
    if (threadIdx.x == 0 && blockIdx.x == 0) g_arena.offset = 0;
}
```

Usage at kernel start:
```cuda
__global__ void recurrent_step_kernel(...) {
    // all transient scratch comes from the arena
    float* trace  = (float*)arena_alloc(hidden * sizeof(float));
    float* delta  = (float*)arena_alloc(hidden * hidden * sizeof(float));
    // ... use trace/delta ...
    // NO free — the arena is reset between frames by a separate 1-thread kernel
}
```

### 2.3 Why this matters for determinism

Deterministic latency requires that no per-frame operation can:
- Block on a system allocator.
- Trigger a page fault on managed memory.
- Stall waiting for a freed buffer to be returned to a pool.

A pre-allocated arena satisfies all three.

### 2.4 Cache coherence (L1 / L2)

The arena should be sized to **fit inside L2** for the steady state. On H100, L2 is 50 MB; on A100, 40 MB; on RTX 4090, 72 MB. If the working set of intermediates for one frame is ≤ L2, every read after the first is served from L2 at ~5–10 TB/s instead of HBM at ~800 GB/s.

Best practice: query the L2 size at startup with `cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev)` and reserve an arena no larger than 60–70% of L2 to leave room for the weights and other persistent data.

### 2.5 `cudaMallocManaged` vs `cudaMalloc + memcpy`

| Option                    | Pros                                         | Cons                                                                |
|---------------------------|----------------------------------------------|---------------------------------------------------------------------|
| `cudaMalloc`              | Predictable; no page faults; fastest on GPU | Requires explicit `cudaMemcpy` to/from host                         |
| `cudaMallocManaged`       | Single pointer, accessible from CPU+GPU     | Page faults on first access can stall 100s of μs; migrate overhead  |
| `cudaMallocHost` (pinned) | Async memcpy possible; no copy on CPU side   | Limited by physical RAM; more expensive to allocate                |
| `cudaMallocAsync` (pool)  | Lower overhead than `cudaMalloc`             | Pool grows monotonically; reset needs explicit `cudaFreeAsync`      |

For our inner loop:
- **Weights, eligibility traces (persistent)**: `cudaMalloc` on device only. Never touch the CPU.
- **Inputs/Outputs to/from the CPU**: `cudaMallocHost` (pinned) for true `cudaMemcpyAsync`.
- **Scratch (per-frame intermediates)**: arena from a single pre-allocated `cudaMalloc` block.
- **Never use `cudaMallocManaged`** inside the hot path.

---

## 3. Latency Micro-benchmarking

### 3.1 The < 50 μs budget — where the time goes

A single fused recurrent-step kernel on H100 with hidden=4096, batch=1, FP16 weights:
- Compute: ~8 GFLOPs → ~8 ms theoretical at 1 PFLOP/s, but on a single SM cluster we get ~10–50 TFLOP/s → ~160–800 μs.
- So our budget assumes **hidden ≤ 1024** or **batch ≤ 32** with vectorized FP16.

The 50 μs budget decomposes as:

```
component                         budget (μs)
kernel launch overhead               3–10
HBM read (inputs)                    5–10
compute (FP16 matmul)               15–25
HBM write (outputs)                  5–10
synchronization / housekeeping       0–5
---------------------------------   ----
total                               ~35–50
```

Below 50 μs, **kernel launch overhead and HBM transfer** dominate, not compute.

### 3.2 CUDA Events for precise timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, shmem, stream>>>(args...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
// ms is the kernel's GPU-side execution time
```

For per-kernel iteration timing:
```cpp
const int N = 1000;
cudaEventRecord(start, stream);
for (int i = 0; i < N; ++i) {
    kernel<<<grid, block, shmem, stream>>>(args...);
}
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
printf("per-iter: %.3f μs\n", ms * 1000.0f / N);
```

### 3.3 Profiling launch overhead vs compute

`cudaEventElapsedTime` measures only the GPU-side time. To measure **end-to-end** (including launch overhead from host side), use `std::chrono::steady_clock`:

```cpp
auto t0 = std::chrono::steady_clock::now();
kernel<<<grid, block, shmem, stream>>>(args...);
cudaStreamSynchronize(stream);  // include sync cost
auto t1 = std::chrono::steady_clock::now();
auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
```

For sub-microsecond precision, use `cudaEventQuery` in a busy-poll loop after `cudaEventRecord(stop, ...)` — this lets you measure completion latency without OS scheduler jitter from `cudaEventSynchronize`.

### 3.4 NVIDIA Nsight Compute

For per-kernel deep-dive, `ncu` (Nsight Compute CLI) is the right tool:

```bash
ncu --set basic \
    --kernel-name recurrent_step_kernel \
    --launch-skip 5 --launch-count 10 \
    --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg,\
dram__bytes_read.sum,dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./bench
```

Key metrics to focus on for our budget:
- `gpu__time_duration.sum` — wall-clock duration of the kernel in ns.
- `sm__cycles_elapsed.avg` — total cycles across all SMs.
- `dram__bytes_read/write.sum` — HBM traffic.
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy.
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — L1 hit rate.

For microsecond-scale work, use the **replay-mode** `--replay-mode kernel` (or `range`) to avoid global clock drift across multiple launches.

### 3.5 Nsight Systems for the timeline

`nsys` gives the wall-clock timeline across all kernels and CPU-side calls:

```bash
nsys profile --stats=true --trace=cuda,nvtx --output=profile ./bench
```

Look for:
- Gaps between kernel launches (host-side overhead).
- Streams out of sync (one stream blocking another).
- Excessive `cudaMalloc` / `cudaFree` on the hot path.

---

## 4. Streaming + Async

### 4.1 `cudaStream_t` for pipelined execution

The default stream (`NULL` / `0`) is **synchronizing** — every call into it implicitly syncs with all other streams. **Never use the default stream for the inner loop.** Always create explicit streams:

```cpp
cudaStream_t compute_stream, copy_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&copy_stream);
```

Then:
```cpp
// Stage 1: copy next input while we compute current
cudaMemcpyAsync(d_input_next, h_input_pinned, input_bytes,
                cudaMemcpyHostToDevice, copy_stream);

// Stage 2: compute on current input
kernel<<<grid, block, shmem, compute_stream>>>(d_input, d_weights, ...);

// Stage 3: event to signal "input next is ready"
cudaEventRecord(input_ready_event, copy_stream);
cudaStreamWaitEvent(compute_stream, input_ready_event, 0);
```

This gives you a 2-stage pipeline: copy(N+1) overlaps compute(N).

### 4.2 `cudaMemcpyAsync` vs synchronous `cudaMemcpy`

**`cudaMemcpy`** blocks the host until the copy is done. It also uses the default stream internally, so it implicitly syncs with all GPU work.

**`cudaMemcpyAsync`** returns immediately on the host. The copy happens on the specified stream.

But — `cudaMemcpyAsync` only achieves **true overlap** when:
1. The host buffer is **pinned** (`cudaMallocHost` or `cudaHostAlloc` with `cudaHostAllocPortable`).
2. The stream is **not the default stream**.

If the host buffer is pageable, the runtime silently does a synchronous copy via a temporary pinned buffer, and you get zero overlap.

### 4.3 Zero host-device sync (critical)

The single biggest source of jitter in CUDA pipelines is **implicit syncs**. Sources:

| Sync source                            | Cost (μs)   | How to eliminate                                  |
|----------------------------------------|-------------|---------------------------------------------------|
| `cudaDeviceSynchronize`                | 10–100+     | Replace with `cudaStreamSynchronize` per stream   |
| `cudaMemcpy` (sync copy)               | 10–100+     | Use `cudaMemcpyAsync` + pinned host mem           |
| Default-stream kernel                  | full queue  | Always explicit stream                            |
| `cudaEventSynchronize` on global event | 5–50        | Use stream events, not global events              |
| `cudaMalloc` / `cudaFree`              | 1–10        | Pre-allocate arena                                |
| Page fault on managed memory           | 100+        | Never use managed memory on hot path              |
| `cudaPointerGetAttributes`             | 1–5         | Cache the result                                  |
| Driver query (e.g. `cudaGetLastError`) | 0.1–1       | Only in debug builds                              |

The discipline: **the host should issue all launches for a frame into streams, then call a single `cudaStreamSynchronize` at the end of the frame**. Nothing else.

### 4.4 CUDA Graphs for repeated launches

For the inner loop, where the same kernel sequence runs every frame with the same shapes, **CUDA Graphs** eliminate launch overhead entirely:

```cpp
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

// Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
kernel_a<<<...>>>(...);
kernel_b<<<...>>>(...);
cudaStreamEndCapture(stream, &graph);

// Instantiate
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

// Replay (launches in ~1 μs regardless of kernel count)
cudaGraphLaunch(graph_exec, stream);
```

Replay cost: ~1 μs total, regardless of how many kernels are in the graph. For our 4-kernel fused path (forward + trace + update + reset), this is a major win.

Source: [CUDA Streams: Asynchronous Execution and Concurrency (abhik.ai)](https://www.abhik.ai/concepts/gpu-computing/cuda-streams)

---

## 5. Fused Recurrent-Layer Kernel Pattern

### 5.1 The four logical phases

For a single recurrent layer with eligibility-trace continual learning:

```
input:  x[B, H]
weight: W[H, H]
state:  h[B, H]
trace:  e[B, H]

Phase 1 (forward):
  h_new = tanh(x @ W + h @ W + b)

Phase 2 (trace update, integrated in forward):
  e_new = decay * e + grad  // grad from local error signal

Phase 3 (weight update, integrated in forward):
  W += lr * e_new.T @ h     // outer product

Phase 4 (plasticity reset, async on second stream):
  if var(e_new) < threshold: e = 0
```

The **forward, trace, and weight update** are all data-parallel over the (B*H) output elements and must execute in one fused kernel. The **plasticity reset** is a tiny reduction — best on a second stream so it can overlap with the next frame's compute.

### 5.2 The fused kernel (skeleton)

```cuda
template<int H, int B>
__global__ void __launch_bounds__(256, 4)   // hint: ≤32 regs/thread, 4 blocks/SM
recurrent_fused_kernel(
    const __half* __restrict__ x,    // [B, H]
    const __half* __restrict__ h,    // [B, H]
    const __half* __restrict__ W,    // [H, H]
    const __half* __restrict__ b,    // [H]
    float*       __restrict__ h_new, // [B, H]
    float*       __restrict__ e,     // [B, H]
    float*       __restrict__ e_new, // [B, H]
    float*       __restrict__ W_new, // [H, H]  (updated in place)
    const float decay,
    const float lr,
    const int   tile = 32)
{
    // Each block handles one (batch, output_chunk) pair.
    const int batch = blockIdx.y;
    const int row0  = blockIdx.x * tile;
    const int tid   = threadIdx.x;
    if (row0 + tid >= H) return;

    // ---- Phase 1: forward (matmul tile) ----
    // Compute W[row0+tid, :] @ x[batch, :] and W[row0+tid, :] @ h[batch, :]
    float acc = __half2float(b[row0 + tid]);
    __shared__ __half x_smem[H];
    __shared__ __half h_smem[H];

    // Cooperative load of x[batch] and h[batch] into shared memory
    for (int i = tid; i < H; i += blockDim.x) {
        x_smem[i] = x[batch * H + i];
        h_smem[i] = h[batch * H + i];
    }
    __syncthreads();

    #pragma unroll 8
    for (int k = 0; k < H; ++k) {
        __half wx = W[(row0 + tid) * H + k];
        acc += __half2float(wx) * __half2float(x_smem[k])
             + __half2float(wx) * __half2float(h_smem[k]);
    }
    h_new[batch * H + row0 + tid] = tanhf(acc);

    // ---- Phase 2: trace update (in registers, no HBM roundtrip) ----
    float trace = e[batch * H + row0 + tid];
    float grad  = /* local error signal */ acc - tanhf(acc);
    float new_trace = decay * trace + grad;
    e_new[batch * H + row0 + tid] = new_trace;

    // ---- Phase 3: weight update (outer product) ----
    // Each thread updates one column of W[row0+tid, :]
    // using the trace value and the pre-loaded x_smem/h_smem.
    #pragma unroll 4
    for (int k = 0; k < H; ++k) {
        float delta = lr * new_trace * __half2float(h_smem[k]);
        // Atomic add is needed because multiple (batch, row) pairs write the same column.
        atomicAdd((float*)&W_new[(row0 + tid) * H + k], delta);
    }
    // NOTE: above uses float atomicAdd on W_new; for FP16 use atomicAdd(__half)
    // from CUDA 11+. If atomic contention is high, switch to a tiled reduction.
}
```

### 5.3 Why `__restrict__` everywhere

`__restrict__` is a **promise** to the compiler that no two `__restrict__`-qualified pointers alias. The compiler then:

1. Keeps values in registers instead of re-reading memory (no aliasing possible).
2. Emits **vectorized** load/store instructions (`LDG.E.128`, `STG.E.128` for `float4`, `int4`).
3. Reorders loads/stores more aggressively.

Without `__restrict__`, the compiler must assume `x`, `h`, and `W` could overlap, and will issue scalar loads with extra `LDG.E` instructions. Vectorized 128-bit loads are typically 4× faster than scalar loads on Ampere/Ada/Hopper for global memory.

### 5.4 Atomic-free weight update (alternative)

The atomic in Phase 3 is a contention hotspot when `B > 1`. Two alternatives:

**Alternative A — per-batch kernel**: launch a separate kernel per batch, no atomics needed:
```cpp
for (int b = 0; b < B; ++b) {
    kernel<<<...>>>(x + b*H, h + b*H, W, b, h_new + b*H, e + b*H);
}
```
Cost: B extra kernel launches. Reasonable for small B.

**Alternative B — reduction in shared memory**: each block computes partial outer products in shared memory, then a single block-level reduction writes to `W_new`:
```cuda
__shared__ float w_delta_smem[TILE][H];
// ... accumulate into w_delta_smem ...
__syncthreads();
if (tid == 0) {
    for (int k = 0; k < H; ++k)
        atomicAdd(&W_new[row0 * H + k], w_delta_smem[tid][k]);
}
```
This reduces B atomics per column to 1, but uses more shared memory.

### 5.5 Async plasticity reset

```cuda
// Separate stream, launches after the fused kernel.
__global__ void plasticity_reset_kernel(
    const float* __restrict__ e_new,
    float*       __restrict__ e,
    int N, float threshold)
{
    // Block-level reduction of variance of e_new
    __shared__ float ssum[32], ssumsq[32];
    float local_sum = 0.f, local_sumsq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = e_new[i];
        local_sum   += v;
        local_sumsq += v * v;
    }
    // warp reduce + block reduce
    // ...
    if (threadIdx.x == 0) {
        float mean = local_sum / N;
        float var  = local_sumsq / N - mean * mean;
        if (var < threshold) {
            // Reset trace
            for (int i = 0; i < N; i += blockDim.x) e[i] = 0.f;
        }
    }
}
```

Launch on `plast_stream` after `cudaEventRecord(fused_done, compute_stream)` and `cudaStreamWaitEvent(plast_stream, fused_done, 0)`. This overlaps the reset with the next frame's compute.

### 5.6 Coherence with `__restrict__`

For vectorization, all pointer arguments to the fused kernel **must** be `__restrict__`-qualified AND aligned to at least 16 bytes (`float4` boundary). CUDA's runtime allocations (`cudaMalloc`) return 256-byte-aligned pointers, so this is satisfied by default. If you allocate via the arena, make sure `arena_alloc` enforces alignment (see §2.2).

---

## 6. Toolchain Setup

### 6.1 `nvcc` flags

```bash
nvcc -O3 -std=c++20 -arch=sm_90 \
     --use_fast_math \
     --maxrregcount=64 \
     --ptxas-options=-v \
     -Xcompiler -Wall,-Wextra,-Wno-unused-parameter \
     -lineinfo \
     -c kernel.cu -o kernel.o
```

Flag rationale:
- `-O3` — highest optimization level.
- `-std=c++20` — needed for `constexpr`, concepts, designated initializers in the host-side launcher.
- `-arch=sm_90` — Hopper (H100). Use `sm_89` for Ada (RTX 4090), `sm_80` for Ampere (A100), `sm_100` for Blackwell B200.
- `--use_fast_math` — enables flush-to-zero, FMA fusion, fast `tanhf`. Acceptable for continual learning.
- `--maxrregcount=64` — caps register usage to preserve occupancy. Tune per kernel via Nsight.
- `--ptxas-options=-v` — prints register usage and spills at compile time.
- `-lineinfo` — keeps line numbers for `cuda-gdb` and Nsight profiling.

### 6.2 Minimal `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.26)
project(agi_continual_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90;89;80")   # Hopper, Ada, Ampere

# nvcc flags applied to all .cu files
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --maxrregcount=64 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wno-unused-parameter")

add_library(agi_kernel STATIC
    src/arena.cu
    src/recurrent_fused.cu
    src/plasticity_reset.cu
)

target_include_directories(agi_kernel PUBLIC include)

enable_testing()
add_executable(bench tests/bench.cu)
target_link_libraries(bench PRIVATE agi_kernel)
add_test(NAME bench COMMAND bench)
```

Source: [CMake FindCUDA module](https://cmake.org/cmake/help/latest/module/FindCUDA.html)

### 6.3 `cuda-gdb` for debug builds

```bash
nvcc -g -G -O0 -std=c++20 -arch=sm_90 -c kernel.cu -o kernel.o
cuda-gdb ./bench
```

Inside `cuda-gdb`:
```
(cuda-gdb) break recurrent_fused_kernel
(cuda-gdb) run
(cuda-gdb) info cuda threads
(cuda-gdb) thread 1  // switch to a specific CUDA thread
(cuda-gdb) print acc
(cuda-gdb) continue
```

`-G` adds device-side debug info. Combine with host `-g` for full source correlation. Debug builds are 5–50× slower and disable most optimizations.

### 6.4 Nsight Systems + Compute

**Nsight Systems** (timeline):
```bash
nsys profile --stats=true --trace=cuda,nvtx --output=profile -f true ./bench
nsys stats profile.nsys-rep
```

**Nsight Compute** (per-kernel deep-dive):
```bash
ncu --set full \
    --kernel-name recurrent_fused_kernel \
    --launch-skip 5 --launch-count 1 \
    -o profile ./bench
```

For automated CI, use `--csv` mode and grep for regressions.

Source: [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

---

## 7. Build & Test Harness

### 7.1 Reproducible benchmark

```cpp
// tests/bench.cu — minimal reproducible harness
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void recurrent_fused_kernel(
    const __half*, const __half*, const __half*, const __half*,
    float*, float*, float*, float*,
    float, float);

int main() {
    constexpr int B = 1, H = 4096;
    constexpr size_t input_bytes  = B * H * sizeof(__half);
    constexpr size_t weight_bytes = H * H * sizeof(__half);

    // ---- pinned host buffers ----
    __half *h_x, *h_W;
    cudaMallocHost(&h_x, input_bytes);
    cudaMallocHost(&h_W, weight_bytes);
    // ... fill with deterministic random data (seed = 42) ...

    // ---- device buffers ----
    __half *d_x, *d_W, *d_b;
    float  *d_h, *d_e, *d_h_new, *d_e_new, *d_W_new;
    cudaMalloc(&d_x, input_bytes);
    cudaMalloc(&d_W, weight_bytes);
    cudaMalloc(&d_b, H * sizeof(__half));
    cudaMalloc(&d_h, B * H * sizeof(float));
    cudaMalloc(&d_e, B * H * sizeof(float));
    cudaMalloc(&d_h_new, B * H * sizeof(float));
    cudaMalloc(&d_e_new, B * H * sizeof(float));
    cudaMalloc(&d_W_new, weight_bytes);

    // ---- warmup ----
    for (int i = 0; i < 20; ++i) {
        recurrent_fused_kernel<<<...>>>(d_x, d_h, d_W, d_b, d_h_new, d_e, d_e_new, d_W_new, 0.9f, 1e-3f);
    }
    cudaDeviceSynchronize();

    // ---- timed ----
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    const int N = 1000;
    cudaEventRecord(s);
    for (int i = 0; i < N; ++i) {
        recurrent_fused_kernel<<<...>>>(d_x, d_h, d_W, d_b, d_h_new, d_e, d_e_new, d_W_new, 0.9f, 1e-3f);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    printf("per-iter: %.3f μs\n", ms * 1000.0f / N);
    return 0;
}
```

### 7.2 Measuring FLOPS and bandwidth utilization

For our kernel:
- FLOPS per iter: `2 * B * H * H` (forward matmul) + `2 * B * H * H` (weight update) = `4 * B * H * H`.
- For B=1, H=4096, FP16: 4 × 1 × 4096² = 67 MFLOPs/iter.
- Achieved TFLOPs/s = `67e6 / (iter_time_s)` / 1e12.

For H100 at ~50 μs/iter: 67 MFLOPs / 50e-6 s = 1.34 TFLOPs/s achieved. H100 peak FP16 is ~197 TFLOPs/s (sparse) or ~99 TFLOPs/s (dense). So we are using ~1.4% of peak — the kernel is **bandwidth-bound or launch-bound**, not compute-bound.

Bandwidth:
- Bytes read per iter: `x + h + W + e = 16 KB + 16 KB + 32 MB + 16 KB ≈ 32 MB`.
- Bytes written per iter: `h_new + e_new + W_new ≈ 32 MB`.
- At 50 μs and HBM at 3 TB/s (H100): `64 MB / 50 μs = 1.28 TB/s` — well within H100 HBM bandwidth.

So our bottleneck at 50 μs is **kernel launch + sync overhead**, not memory or compute. This is the right diagnosis to make before optimizing.

### 7.3 Correctness validation against CPU reference

```cpp
// Reference impl on CPU (same algorithm, naive)
void recurrent_step_cpu(
    const float* x, const float* h, const float* W, const float* b,
    float* h_new, float* e, float* e_new, float* W_new,
    int B, int H, float decay, float lr)
{
    for (int bi = 0; bi < B; ++bi) {
        for (int i = 0; i < H; ++i) {
            float acc = b[i];
            for (int j = 0; j < H; ++j) {
                acc += W[i*H + j] * (x[bi*H + j] + h[bi*H + j]);
            }
            h_new[bi*H + i] = tanhf(acc);
            float grad = acc - tanhf(acc);
            e_new[bi*H + i] = decay * e[bi*H + i] + grad;
            for (int j = 0; j < H; ++j) {
                W_new[i*H + j] += lr * e_new[bi*H + i] * h[bi*H + j];
            }
        }
    }
}

bool validate(const float* gpu_out, const float* cpu_out, int N, float eps = 1e-3f) {
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        float d = fabsf(gpu_out[i] - cpu_out[i]);
        float m = fmaxf(fabsf(gpu_out[i]), fabsf(cpu_out[i]));
        if (d > eps * fmaxf(m, 1e-6f)) {
            if (++bad < 10) printf("MISMATCH @ %d: gpu=%.6f cpu=%.6f\n", i, gpu_out[i], cpu_out[i]);
        }
    }
    return bad == 0;
}
```

Tolerance: `1e-3` absolute or `1e-3 * max(|a|, |b|)` relative. Tighter than this for FP16 typically fails due to non-associativity of floating-point.

---

## 8. Classic CUDA Pitfalls

### 8.1 Bank conflicts in shared memory

Shared memory is organized as **32 banks** of 4 bytes each. A bank conflict occurs when two threads in a warp access the **same bank** but **different addresses** in the same cycle — the access is serialized.

```cuda
// BAD: stride-1 on a 2D tile of width 33 — all threads hit the same bank
__shared__ float tile[32][33];
float v = tile[tid][col];   // bank conflict on every load

// GOOD: pad to width 33, or use stride 32 (no conflict)
__shared__ float tile[32][32];   // but now writes to tile[k][tid] in a loop
                                //   hit bank 0 on every iteration
// BEST: tile[32][32+1] padding breaks the alignment
__shared__ float tile[32][33];
```

The **broadcast exception**: multiple threads reading the **same address** in the same bank is a multicast (no conflict). Useful for loading constants.

Source: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 8.2 Unaligned memory access

A 32-byte aligned `float` array accessed by 32 threads in a warp = 1 sector = optimal.

```cuda
// BAD: misaligned base pointer (offset 1 byte from 32-byte boundary)
const float* p = ((const float*)base) + 1;   // unaligned
float v = p[tid];   // compiler may emit 4 sectors instead of 1

// GOOD: enforce alignment at allocation
float* p;
cudaMalloc(&p, N * sizeof(float));   // always 256-byte aligned
```

Always pass `__restrict__` and ensure 16-byte alignment (`float4` boundary) to enable vectorized loads.

Source: [How to Access Global Memory Efficiently in CUDA C/C++](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-cpp/)

### 8.3 Register spilling

Detected by `--ptxas-options=-v`:
```
ptxas info    : Used 64 registers, 16 bytes cmem[0], 8 bytes smem
ptxas info    : Used 80 registers, 16 bytes cmem[0], 8 bytes smem
                ^^^^^ no spill yet
ptxas info    : Used 80 registers, storing 16 bytes in local memory
                ^^^^^ 16 bytes spilled to local memory (cached in L1)
```

Mitigations:
1. Reduce live values (recompute instead of store).
2. Tile the work.
3. Lower precision intermediates (FP16 → FP32 cast at use site).
4. Use `#pragma unroll` selectively — over-unrolling increases register pressure.

### 8.4 Implicit syncs at kernel boundaries

Every kernel launch is a synchronization point **within a stream** — the next kernel in the same stream waits for the previous to complete. But across streams, no sync is implied.

```cuda
// Kernel A launches on stream 1
kernel_A<<<..., stream_1>>>(...);
// Kernel B launches on stream 2 — does NOT wait for A
kernel_B<<<..., stream_2>>>(...);

// To make B wait for A:
cudaEvent_t e;
cudaEventRecord(e, stream_1);
cudaStreamWaitEvent(stream_2, e, 0);
kernel_B<<<..., stream_2>>>(...);
```

Other implicit sync sources:
- `cudaMemcpy` between device buffers: synchronous on the default stream.
- Reading from a managed-memory pointer on the CPU side: full device sync.
- Any `cudaGetLastError()` call after a launch: serializes.

### 8.5 Other gotchas

| Pitfall                                   | Symptom                              | Fix                                                  |
|-------------------------------------------|--------------------------------------|------------------------------------------------------|
| Warp divergence (`if (tid % 2) ...`)      | 50% perf on conditionals             | Restructure so each warp takes one branch            |
| Excessive `__syncthreads()`               | Stalls, low occupancy                | Use warp-level `__syncwarp()` where possible         |
| Passing structs by value                  | Spills registers                     | Pass by pointer or `__restrict__` reference           |
| `#pragma unroll` on huge loops            | Massive register pressure            | Cap with `#pragma unroll N`                          |
| `cudaMallocManaged` on hot path           | 100+ μs page-fault stalls            | Use `cudaMalloc` + explicit copies                   |
| Page-locked host memory not used          | Async memcpy falls back to sync      | Always `cudaMallocHost` for buffers touched by async |
| Default stream (NULL) for async work      | Implicit sync, no overlap            | Always create and use explicit streams               |
| Float atomics on shared memory            | Slow (no HW atomic for shared)       | Use `atomicAdd_system` or restructure               |

---

## Appendix A — Sources

### Papers
- **FlashAttention** (Dao et al., 2022) — original fused attention kernel.
- **FlashAttention-2** (Dao, 2023) — improved parallelism. [OpenReview](https://openreview.net/forum?id=mZn2Xyh9Ec)
- **FlashAttention-4** (Dao et al., 2026) — algorithm / kernel pipelining co-design. [arXiv](https://arxiv.org/html/2603.05451v1)
- **vLLM PagedAttention** (Kwon et al., SOSP '23) — fused KV-cache attention with paging.

### NVIDIA official documentation
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### NVIDIA developer blog
- [CUDA Refresher: Memory Coalescing](https://developer.nvidia.com/blog/cuda-refresher-the-cuda-c-programming-model-and-memory-coalescing/)
- [How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-cpp/)
- [NVIDIA Developer Blog home](https://developer.nvidia.com/blog/)

### Community / third-party
- [Aman's AI Journal — FlashAttention primer](https://aman.ai/primers/ai/flashattention/)
- [Hugging Face blog — FlashAttention](https://huggingface.co/blog/garg-aayush/flash-attention)
- [CUDA Streams guide (abhik.ai)](https://www.abhik.ai/concepts/gpu-computing/cuda-streams)
- [CUDA 01 — Streams & Events walkthrough (Medium)](https://yc-kuo.medium.com/cuda-01-streams-events-walkthrough-5ff0a32fc1ea)

### Memory allocator references (Quake / Carmack heritage)
- [Quake Arena Allocator notes (jfo/codearchitect)](https://github.com/jfo/codearchitect/blob/master/notes/Quake%20Arena%20Allocator.md)
- [A Quick Look at the Quake Source Code (fabiensanglard.net)](https://fabiensanglard.net/quake/index.php)
- [Memory Management: A Look at Quake 3 (fabiensanglard.net)](https://fabiensanglard.net/quake3/memory_management.php)

### CMake / build
- [CMake FindCUDA module](https://cmake.org/cmake/help/latest/module/FindCUDA.html)
- [CUDA 12 C++20 with CMake (Stack Overflow)](https://stackoverflow.com/questions/75010206/cuda-12-c20-support-with-cmake-not-working)
- [Passing flags to nvcc via CMake (NVIDIA forums)](https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768)

---

## Appendix B — Quick Checklist for a New Kernel

Before declaring a fused kernel "done", verify:

- [ ] Register usage ≤ 64 per thread (`--ptxas-options=-v`)
- [ ] No local-memory spills (`ptxas` output clean)
- [ ] Shared-memory accesses have no bank conflicts (Nsight Compute `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` = 0)
- [ ] All global memory accesses are coalesced (Nsight `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` near peak)
- [ ] All pointer arguments `__restrict__`-qualified
- [ ] All allocations 16-byte aligned minimum (256-byte for safety)
- [ ] Kernel launched on a non-default stream
- [ ] Per-frame timing < 50 μs end-to-end on target GPU
- [ ] HBM bandwidth utilization > 60% of peak (proves we are bandwidth-bound, not launch-bound)
- [ ] Correctness validated against CPU reference within tolerance
- [ ] No `cudaMalloc` / `cudaFree` / `cudaDeviceSynchronize` in the inner loop
- [ ] Replay-mode Nsight Compute profile attached to the kernel
- [ ] Test runs deterministically (seeded inputs, no `rand()`)
