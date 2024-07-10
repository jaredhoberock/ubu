#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/detail/to_integral_like.hpp"

namespace ubu::detail
{

template<coordinate C1, congruent<C1> C2>
constexpr congruent<C1> auto coordinate_max(const C1& coord1, const C2& coord2)
{
  if constexpr (scalar_coordinate<C1>)
  {
    return to_integral_like(coord1) < to_integral_like(coord2) ? to_integral_like(coord2) : to_integral_like(coord1);
  }
  else
  {
    return tuples::zip_with(coord1, coord2, [](const auto& c1, const auto& c2)
    {
      return coordinate_max(c1, c2);
    });
  }
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

