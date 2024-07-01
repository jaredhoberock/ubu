#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../../miscellaneous/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "detail/to_integral_like.hpp"


namespace ubu
{


template<scalar_coordinate C1, scalar_coordinate C2>
constexpr integral_like auto coordinate_difference(const C1& coord1, const C2& coord2)
{
  return detail::to_integral_like(coord1) - detail::to_integral_like(coord2);
}

template<nonscalar_coordinate C1, nonscalar_coordinate C2>
  requires congruent<C1,C2>
constexpr congruent<C1> auto coordinate_difference(const C1& coord1, const C2& coord2)
{
  return tuples::zip_with(coord1, coord2, [](const auto& c1, const auto& c2)
  {
    return coordinate_difference(c1, c2);
  });
}


} // end ubu

#include "../../detail/epilogue.hpp"

