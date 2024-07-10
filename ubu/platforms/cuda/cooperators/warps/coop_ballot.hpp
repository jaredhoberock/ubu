#pragma once

#include "../../../../detail/prologue.hpp"

#include "warp_like.hpp"
#include <cstddef>

namespace ubu::cuda
{


template<warp_like W>
constexpr std::uint32_t coop_ballot(W, bool value)
{
#if defined(__CUDACC__)
  return __ballot_sync(warp_mask, value);
#else
  return 0;
#endif
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"


