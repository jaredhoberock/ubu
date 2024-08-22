#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "detail/to_integral_like.hpp"


namespace ubu
{


template<coordinate A, coordinate B>
  requires congruent<A,B>
constexpr congruent<A> auto coordinate_difference(const A& a, const B& b)
{
  if constexpr (unary_coordinate<A>)
  {
    return detail::to_integral_like(a) - detail::to_integral_like(b);
  }
  else
  {
    return tuples::zip_with(a, b, [](const auto& a_i, const auto& b_i)
    {
      return coordinate_difference(a_i, b_i);
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

