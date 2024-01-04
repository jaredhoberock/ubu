#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/coordinate.hpp"
#include <utility>

namespace ubu::detail
{

template<scalar_coordinate C>
constexpr auto as_integral(const C& coord)
{
  if constexpr (std::integral<C>) return coord;
  else return get<0>(coord);
}

template<scalar_coordinate C>
using as_integral_t = decltype(as_integral(std::declval<C>()));

} // end ubu::detail

#include "../../../detail/epilogue.hpp"

