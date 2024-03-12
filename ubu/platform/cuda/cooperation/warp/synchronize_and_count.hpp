#pragma once

#include "../../../../detail/prologue.hpp"

#include "warp_like.hpp"

namespace ubu::cuda
{


template<warp_like W>
constexpr int synchronize_and_count(W, bool value)
{
#if defined(__CUDACC__)
  return __popc(__ballot_sync(warp_mask, value));
#else
  return -1;
#endif
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

