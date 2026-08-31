# GuimLab Studio — Vendored Third-Party Dependencies & Provenance

This document tracks the provenance, upstream repositories, tags, and license declarations for all vendored graphics and UI libraries used in `guim_studio`.

---

## 1. Dear ImGui
- **Upstream Repository**: https://github.com/ocornut/imgui
- **Vendored Version**: `v1.93.0 WIP`
- **Location**: `src/studio/third_party/imgui/`
- **License**: MIT License (Copyright (c) 2014-2026 Omar Cornut)
- **Included Files**:
  - `imconfig.h`
  - `imgui.h`
  - `imgui_internal.h`
  - `imgui.cpp`
  - `imgui_draw.cpp`
  - `imgui_tables.cpp`
  - `imgui_widgets.cpp`
  - `imstb_rectpack.h`
  - `imstb_textedit.h`
  - `imstb_truetype.h`
- **Modifications**: None (standard clean upstream headers and implementation).

---

## 2. ImPlot
- **Upstream Repository**: https://github.com/epezent/implot / https://github.com/brenocq/implot
- **Vendored Version**: `v1.1 WIP` (2025-2026 active maintenance tree)
- **Location**: `src/studio/third_party/implot/`
- **License**: MIT License (Copyright (c) 2020-2024 Evan Pezent, Copyright (c) 2025-2026 Breno Cunha Queiroz)
- **Included Files**:
  - `implot.h`
  - `implot_internal.h`
  - `implot.cpp`
  - `implot_items.cpp`
- **Modifications**: None (pure upstream math and vector plotting routines).

---

## 3. Sokol Header-Only Cross-Platform Graphics
- **Upstream Repository**: https://github.com/floooh/sokol
- **Vendored Version**: Latest 2026 stable header-only release
- **Location**: `include/sokol/`
- **License**: zlib/libpng license (Copyright (c) 2018 Andre Weissflog)
- **Included Files**:
  - `sokol_app.h`
  - `sokol_gfx.h`
  - `sokol_imgui.h`
  - `sokol_glue.h`
- **Modifications**: None.

---

## Security & Conformance Guarantee
- All vendored libraries are statically compiled without external network dependencies.
- Zero dynamic allocations are made inside real-time render loops.
- All format string invocations are bounded to static string literals.
