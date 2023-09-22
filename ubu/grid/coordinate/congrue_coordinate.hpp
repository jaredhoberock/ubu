#pragma once

#include "../../detail/prologue.hpp"

#include "coordinate.hpp"
#include "detail/coordinate_inclusive_scan.hpp"
#include <utility>


namespace ubu
{


template<coordinate C1, coordinate C2>
  requires weakly_congruent<C1,C2>
constexpr congruent<C2> auto congrue_coordinate(const C1& coord, const C2& shape)
{
  // congrue_coordinate is essentially a divmod operation
  // for each element of coord, we want to apply a divmod to the corresponding element of shape
  // that mode's result is the remainder of that operation, and we pass the quotient of the division
  // "to the right" as the carry
  //
  // congrue_coordinate's final_combine operation ignores the carry and returns its first parameter

  // the combine operation is divmod
  // the result of the operation is the remainder, and the "carry" is the quotient
  auto combine = [](auto current_divisor, auto prev_quotient)
  {
    auto quotient = prev_quotient / current_divisor;
    auto remainder = prev_quotient % current_divisor;
    return std::pair(remainder, quotient);
  };

  // the final combine operation ignores the divisor and returns the previous quotient
  auto final_combine = [](auto current_divisor, auto prev_quotient)
  {
    return prev_quotient;
  };

  return detail::coordinate_inclusive_scan_with_final(shape, coord, combine, final_combine);
}


} // end ubu

#include "../../detail/epilogue.hpp"

