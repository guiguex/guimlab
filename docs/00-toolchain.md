# 00 — Toolchain

Everything you need to install before `cmake --build` will succeed on `app/`.

## 1. Prerequisites

| Component            | Minimum version | Recommended | Notes                                 |
|----------------------|-----------------|-------------|----------------------------------------|
| **CUDA Toolkit**     | 12.0            | 12.6+       | `nvcc` on PATH                          |
| **GPU**              | Volta (sm_70)   | Ampere/Ada (sm_80+) | RTX 3060+ recommended        |
| **CMake**            | 3.22            | 3.27+       | multi-config on Windows needs 3.20+  |
| **Host compiler**    | MSVC 19.30 / GCC 11 / Clang 15 | MSVC 2022 17.6 / GCC 13 | C++20 required |
| **Python** (opt.)    | 3.10            | 3.12        | only needed for some helper scripts   |
| **Ninja** (opt.)     | 1.11            | 1.12+       | dramatically faster than Make on Linux |

> ⚠️ **Compute capability < 7.0** (Pascal, Maxwell) — the kernel still compiles
> (with a warning) but you must override `GUIM_CUDA_ARCH` and performance will
> be poor. Prefer Volta (V100, Titan V) or newer.

---

## 2. Installation

### 2.1 Windows

1. **Visual Studio 2022** — Community edition is fine.
   - Workload: *Desktop development with C++*.
   - Components: *MSVC v143*, *Windows 11 SDK*, *C++ CMake tools for Windows*.

2. **CUDA Toolkit 12.x** — download from
   <https://developer.nvidia.com/cuda-downloads>.
   - Pick *Windows → x86_64 → 11 → exe (local)*.
   - Default install puts `nvcc` at
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X\bin\nvcc.exe`.
   - Add that `bin\` to `PATH` *and* set
     `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X`.

3. **CMake** — either via VS Installer ("CMake tools for Windows") or
   <https://cmake.org/download/>.

4. **NVIDIA Driver** — matched to the CUDA minor version you installed.
   <https://www.nvidia.com/Download/index.aspx>.

5. **Verify**:

   ```bat
   nvcc --version
   nvidia-smi
   ```

### 2.2 Linux (Ubuntu 22.04 / 24.04)

```bash
# CUDA 12.6 — see NVIDIA's "runfile" or repo install docs:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

sudo apt update
sudo apt install -y cmake ninja-build g++-13

# Verify
nvcc --version
nvidia-smi
```

### 2.3 macOS

CUDA is **no longer supported on macOS** in any recent toolchain. Develop on
Linux/Windows or use a remote GPU box. The CPU reference in `src/guim_cpu_ref.cpp`
still builds with Apple Clang for testing math logic.

---

## 3. First build

```bash
cd app
bash scripts/check_gpu.sh     # sanity check
bash scripts/build.sh         # Release build + auto-bench
```

Expected output of `check_gpu.sh`:

```
nvcc version:        12.6
Driver version:      555.42
GPU:                 NVIDIA GeForce RTX 4070
Compute capability:  8.9
SM count:            46
Total memory:        12282 MiB
```

If you see anything else, jump to **§ 5 Troubleshooting** below.

---

## 4. Common CMake variables

```bash
# pick a specific arch (skip the defaults)
cmake -B build -S . -DGUIM_CUDA_ARCH="80;86;89"

# debug build with sanitizers
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DGUIM_ENABLE_ASAN=ON

# verbose (one -v per make job)
cmake --build build -j --verbose
```

`compile_commands.json` is emitted at `build/compile_commands.json` — point
clangd / VSCode / Zed at it for IDE support.

---

## 5. Troubleshooting

### 5.1 `nvcc not found`

- **Linux**: ensure `/usr/local/cuda/bin` is on `PATH`, or `source` the
  profile script:
  `source /etc/profile.d/cuda.sh`.
- **Windows**: add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X\bin`
  to your user PATH, restart your shell.
- After install, reopen the terminal — `set PATH` is not propagated to new
  shells.

### 5.2 `No CMAKE_CUDA_COMPILER could be found`

CMake couldn't locate the CUDA compiler. Two fixes:

```bash
# Linux/macOS
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -S .

# Windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X
cmake -B build -S .
```

### 5.3 `Unsupported gpu architecture 'compute_XX'`

Your driver is older than the arch CMake is trying to compile for. Either:

- update the driver (matches your CUDA toolkit), or
- drop the offending arch from `GUIM_CUDA_ARCH`, e.g. `-DGUIM_CUDA_ARCH="70;80"`.

### 5.4 `MSB3721` on Windows ("The command exited with code %errorlevel%")

The Visual Studio generator picked a stale CUDA target. Two cures:

1. **Right-click the project → Properties → CUDA C/C++ → Target Machine**:
   set *64-bit (x64)*.
2. Or switch to Ninja: `cmake -B build -S . -G Ninja`.

### 5.5 `error: identifier "thrust" is undefined`

Means the `Thrust::Thrust` target wasn't found. On CUDA 12 this should be
automatic; if you have multiple CUDA installs, set `CUDACXX` and re-configure
*with a fresh build directory* (`rm -rf build` first — CMake caches
detected paths aggressively).

### 5.6 `Cannot find -lcudart`

Linux only. Install the runtime libs explicitly:

```bash
# Debian/Ubuntu
sudo apt install -y cuda-cudart-dev-12-6

# or, if you used the runfile installer:
sudo ldconfig /usr/local/cuda/lib64
```

### 5.7 Bench reports 0 fps

You have no usable GPU (driver missing, MIG mode active, container without
`--gpus all`). Re-run `check_gpu.sh` first — it'll fail with the actual reason.

### 5.8 Tests fail with `CUDART error: out of memory`

Reduce `GUIM_STATE_DIM` × `GUIM_TOTAL_DIM` allocation or free GPU memory from
other processes (`nvidia-smi` shows who's using it). The staging impl
allocates ~4× the minimum arena.

### 5.9 Slow first build

GoogleTest is fetched and built from source by `FetchContent` (~30 s).
Subsequent builds are incremental.

---

## 6. Updating the toolchain

```bash
# CUDA
sudo apt install -y cuda-toolkit-12-6   # Linux
winget install NVIDIA.CUDA              # Windows (12.x stable)

# CMake
sudo apt upgrade cmake                  # Linux
winget upgrade Kitware.CMake            # Windows
```

Then `rm -rf build && bash scripts/build.sh` to force a clean reconfigure.

---

## 7. CI sanity

For CI runners without a GPU, the kernel still **compiles** (we use
`-allow-unsupported-compiler` and the toolchain doesn't run the binary). To
skip GPU-dependent steps:

```bash
cmake -B build -S . -DGUIM_BUILD_TESTS=OFF -DGUIM_BUILD_BENCH=OFF
cmake --build build -j
```

Tests and the bench executable will not be produced, but `libguim.a` /
`guim.lib` will be, so downstream consumers that link statically are happy.

---

## 8. References

- CUDA Toolkit archive: <https://developer.nvidia.com/cuda-toolkit-archive>
- CUDA programming guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- Thrust: <https://nvidia.github.io/cccl/thrust/>
- GoogleTest: <https://google.github.io/googletest/>
- CMAKE_CUDA_ARCHITECTURES: <https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html>