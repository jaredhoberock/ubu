#pragma once

#include "../../../../detail/prologue.hpp"

#include "block_like.hpp"

namespace ubu::cuda
{

template<block_like B>
constexpr int synchronize_and_count(B, bool value)
{
#if defined(__CUDACC__)
  return __syncthreads_count(value);
#else
  return -1;
#endif
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

