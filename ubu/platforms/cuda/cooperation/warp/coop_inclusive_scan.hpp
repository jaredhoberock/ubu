#pragma once

#include "../../../../detail/prologue.hpp"

#include "shuffle_up.hpp"
#include "warp_like.hpp"
#include <concepts>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{


template<warp_like W, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr T coop_inclusive_scan(W self, T value, F binary_op)
{
  for(int offset = 1; offset <= warp_size/2; offset *= 2)
  {
    T other = shuffle_up(self, offset, value);
  
    if(id(self) >= offset)
    {
      value = binary_op(value, other);
    }
  }

  return value;
}


template<warp_like W, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr std::optional<T> coop_inclusive_scan(W self, std::optional<T> value, F binary_op)
{
  int num_values = synchronize_and_count(self, value.has_value());

  if(num_values == warp_size)
  {
    return coop_inclusive_scan(self, *value, binary_op);
  }

  return coop_inclusive_scan(self, value, [binary_op](std::optional<T> x, std::optional<T> y)
  {
    if(x and y) return std::make_optional(binary_op(*x,*y));
    else if(x) return x;
    return y;
  });
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

