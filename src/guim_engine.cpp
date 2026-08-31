// =============================================================================
//  guim_engine.cpp  -  Host-side EngineImpl shim.
//  =============================================================================
//  Author        : Guimlab Contributors
//  SPDX-License   : MIT
// =============================================================================

#include "guim.h"

namespace guim {

Engine make_engine(std::uint32_t seed) noexcept
{
    return Engine{seed};
}

}  // namespace guim