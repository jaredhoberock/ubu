#pragma once

#include "../../../../detail/prologue.hpp"

#include "coop_inclusive_scan.hpp"
#include "shuffle_up.hpp"
#include "warp_like.hpp"
#include <concepts>
#include <type_traits>

namespace ubu::cuda
{


template<warp_like W, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr T coop_exclusive_scan(W self, T init, T value, F binary_op)
{
  if(id(self) == 0)
  {
    value = binary_op(init, value);
  }

  value = coop_inclusive_scan(self, value, binary_op);

  value = shuffle_up(self, value, 1);
  if(id(self) == 0)
  {
    value = init;
  }

  return value;
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

