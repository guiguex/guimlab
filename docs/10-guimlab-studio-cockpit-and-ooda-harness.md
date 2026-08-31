# GuimLab Studio & neuromorphic Development Harness (Cockpit Architecture)

**Author:** GuimLab Research & Engineering  
**Version:** 0.3.0-NextGen  
**License:** AGPL-3.0 / MIT  

---

## 1. Overview & Architectural Isolation Doctrine

`guim_studio` is a 1000-FPS pure C++20 / Dear ImGui / ImPlot visual cockpit and telemetry fabric designed for the GuimLab continuous-time neuromorphic neuromorphic engine.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GUIMLAB ENGINE (CUDA BARE-METAL)                      │
│   Reflex L0 (<100 ns)  │  Lovelace Cortex (<5 µs)  │  CBP Plasticity Sweeper│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Lock-Free POSIX/Win32 SHM (Zero-Copy)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GUIMLAB STUDIO VISUAL COCKPIT                       │
│  [Thermal Cortex Map] [SR-MIT Phase Space] [16kHz Audio] [TMD-ET] [Telemetry]│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Zero-Pollution Invariants:
1. **Decoupled Process Architecture**: `guim_studio` runs as an independent visualization process. It polls the shared memory ring buffers asynchronously. The core GPU inference loop never makes blocking calls to the GUI.
2. **Zero Dynamic Allocation in GUI Loop**: All plotting series, audio oscilloscopes, and latency trackers utilize pre-allocated static circular buffers (`guim::studio::CircularBuffer`).
3. **High-Frequency Ingestion**: The studio runs at up to 1000 FPS (or vsync-locked), providing instantaneous visual feedback on membrane potential evolution, dead neuron detection, and symplectic phase orbits.

---

## 2. Specialized Visual Modules

### A. Lovelace Cortex 256-Neuron Dynamic Thermal Heatmap
* **Source**: `src/studio/panels/cortex_thermal_map.cpp`
* **Features**:
  * 16x16 matrix representation of cortical activations $h_i(t)$.
  * Real-time metabolic variance tracking $\sigma_i^2 = \beta_{ema} \sigma_i^2 + (1-\beta_{ema})(h_i - \mu_i)^2$.
  * Visual alert for dead units ($\sigma_i^2 < \epsilon_{plasticity}$) triggering asynchronous CBP neurogenesis.

### B. SR-MIT Symplectic Riemannian Phase-Space Orbit
* **Source**: `src/studio/panels/symplectic_phase_portrait.cpp`
* **Features**:
  * 2-form phase space $(E_{ij}, P_{ij}) \in \mathbb{R}^2$ trajectory plotting.
  * Real-time verification of the Kuramoto harmonic phase-lead compensation $\gamma(\omega_{ij}, \tau) = \frac{\omega_{ij} \tau}{\sqrt{1 + (\omega_{ij}\tau)^2}}$.
  * Immediate visual detection of harmonic phase lag elimination.

### C. 16 kHz Streaming Audio & DDWR Sensory Scope
* **Source**: `src/studio/panels/audio_oscilloscope.cpp`
* **Features**:
  * Real-time display of the 128 INT8 packed acoustic formant stream.
  * Live Delta-Driven Warp Routing (DDWR) sparsity gauge displaying VRAM bandwidth reduction percentage (typically $>95\%$).

### D. TMD-ET Meta-Gradients & Attention Barometer
* **Source**: `src/studio/panels/metagradients_inspector.cpp`
* **Features**:
  * Reflex L0 16-channel motor distribution monitor.
  * Per-synapse adaptive meta-learning rate tracker $\alpha_{ij} = \exp(\beta_{ij})$.
  * Neuromodulator chemical state indicators (Dopamine $\delta(t)$, Serotonin).

### E. Hardware Latency Profiler & VRAM Sentinel
* **Source**: `src/studio/panels/telemetry_cockpit.cpp`
* **Features**:
  * Kernel latency gauge ($3.8\,\mu\text{s}$) and host-to-device round-trip latency histogram ($p_{50}, p_{99}$).
  * Interactive chemical stimulation: Live sliders and buttons to inject Dopamine and Serotonin directly into the running GPU memory without stopping execution.

---

## 3. How to Build and Run

### Compilation
```bash
# Configure with Studio enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGUIM_BUILD_STUDIO=ON

# Build the studio executable
cmake --build build --target guim_studio -j
```

### Launching the Full Stack
```bash
# Terminal 1: Launch the persistent dual-speed neuromorphic node
./build/bin/guim_node_v3

# Terminal 2: Launch the visual cockpit
./build/bin/guim_studio
```
