#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../detail/reflection/is_device.hpp"
#include "warp_like.hpp"
#include <bit>

namespace ubu::cuda
{

template<warp_like W>
constexpr bool synchronize_and_any(W warp, bool value)
{
#if defined(__CUDACC__)
  if UBU_TARGET(ubu::detail::is_device())
  {
    return __any_sync(warp_mask, value);
  }
  else
  {
    return false;
  }
#else
  return false;
#endif
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

