#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/detail/as_integral.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"

namespace ubu::detail
{

template<coordinate C1, congruent<C1> C2>
constexpr congruent<C1> auto coordinate_max(const C1& coord1, const C2& coord2)
{
  if constexpr (scalar_coordinate<C1>)
  {
    return as_integral(coord1) < as_integral(coord2) ? as_integral(coord2) : as_integral(coord1);
  }
  else
  {
    return tuple_zip_with(coord1, coord2, [](const auto& c1, const auto& c2)
    {
      return coordinate_max(c1, c2);
    });
  }
}

} // end ubu::detail

#include "../../detail/prologue.hpp"

