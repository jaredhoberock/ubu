#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/constant_valued.hpp"

#include <concepts>

namespace ubu::detail
{

template<class T>
concept constant_valued_integral =
  constant_valued<T>
  and std::integral<constant_value_t<T>>
;

// as_integral converts a type into an integral
template<class I>
  requires (std::integral<I> or constant_valued_integral<I>)
constexpr std::integral auto as_integral(const I& i)
{
  if constexpr (std::integral<I>)
  {
    // case 0: I is already std::integral
    return i;
  }
  else
  {
    // case 1: constant_value_v<I> is std::integral
    return constant_value_v<I>;
  }
}

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

