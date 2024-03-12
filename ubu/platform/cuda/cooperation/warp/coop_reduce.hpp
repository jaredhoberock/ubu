#pragma once

#include "../../../../detail/prologue.hpp"

#include "shuffle_down.hpp"
#include "synchronize_and_count.hpp"
#include "warp_like.hpp"
#include <concepts>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{


template<warp_like W, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>, T>)
constexpr std::optional<T> coop_reduce(W self, std::optional<T> value, F binary_op)
{
  int num_values = synchronize_and_count(self, value.has_value());

  auto num_threads = warp_size;
  auto num_passes = 5_c; // this is log2(warp_size)

  if(num_values == num_threads)
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = shuffle_down(self, *value, offset);
      value = binary_op(*value, other);
    }
  }
  else
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      std::optional other = shuffle_down(self, value, offset);
      if((id(self) + offset < num_values) and other) *value = binary_op(*value, *other);
    }
  }

  return (id(self) == 0) ? value : std::nullopt;
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

