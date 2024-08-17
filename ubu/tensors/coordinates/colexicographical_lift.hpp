#pragma once

#include "../../detail/prologue.hpp"

#include "concepts/coordinate.hpp"
#include "detail/coordinate_inclusive_scan.hpp"
#include "traits/zeros.hpp"
#include <utility>


namespace ubu
{

// colexicographical_lift "upcasts" a weakly_congruent coordinate into a higher-dimensional space described by a shape
// because this lift operation is colexicographical, it "aligns" the modes of coord and shape at the left, and proceeds from left to right
// when the coordinate is congruent with the shape, colexicographical_lift is the identity function

template<coordinate C1, coordinate C2>
  requires weakly_congruent<C1,C2>
constexpr congruent<C2> auto colexicographical_lift(const C1& coord, const C2& shape)
{
  // colexicographical_lift is essentially a divmod operation
  // for each element of coord, we want to apply a divmod to the corresponding element of shape
  // that mode's result is the remainder of that operation, and we pass the quotient of the division
  // "to the right" as the carry
  //
  // colexicographical_lift's final_combine operation ignores the divisor and returns the previous carry
  
  // the combine operation is divmod
  // the result of the operation is the remainder, and the "carry" is the quotient
  auto combine = [](auto prev_quotient, auto current_divisor)
  {
    auto quotient = prev_quotient / current_divisor;
    auto remainder = prev_quotient % current_divisor;
    return std::pair(remainder, quotient);
  };
  
  // the final combine operation ignores the divisor and returns the previous quotient
  auto final_combine = [](auto prev_quotient, auto current_divisor)
  {
    return prev_quotient;
  };
  
  return detail::coordinate_inclusive_scan_with_final(shape, coord, combine, final_combine);
}


} // end ubu

#include "../../detail/epilogue.hpp"

