#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/integral/integral_like.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../coordinate_cast.hpp"
#include "../colexicographical_lift.hpp"
#include "../coordinate_sum.hpp"
#include <concepts>


namespace ubu
{

template<coordinate C, congruent<C> S, integral_like I>
constexpr void colexicographical_advance(C& coord, const S& shape, I n)
{
  congruent<C> auto delta = colexicographical_lift(n, shape);
  coord = coordinate_cast<C>(coordinate_sum(coord, delta));
}


} // end ubu


#include "../../../detail/epilogue.hpp"

