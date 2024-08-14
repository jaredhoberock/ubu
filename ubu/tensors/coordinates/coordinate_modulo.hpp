#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "detail/to_integral_like.hpp"

namespace ubu
{

template<coordinate C1, coordinate C2>
  requires congruent<C1,C2>
constexpr congruent<C1> auto coordinate_modulo(const C1& dividend, const C2& divisor)
{
  if constexpr (unary_coordinate<C1>)
  {
    return detail::to_integral_like(dividend) % detail::to_integral_like(divisor);
  }
  else
  {
    return tuples::zip_with(dividend, divisor, [](const auto& a, const auto& b)
    {
      return coordinate_modulo(a, b);
    });
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

