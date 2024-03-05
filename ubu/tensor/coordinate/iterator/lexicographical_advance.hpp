#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../concepts/integral_like.hpp"
#include "../coordinate_cast.hpp"
#include "../math/coordinate_sum.hpp"
#include "../lexicographical_lift.hpp"


namespace ubu
{

template<coordinate C, congruent<C> S, integral_like I>
constexpr void lexicographical_advance(C& coord, const S& shape, I n)
{
  congruent<C> auto delta = lexicographical_lift(n, shape);
  coord = coordinate_cast<C>(coordinate_sum(coord, delta));
}


} // end ubu


#include "../../../detail/epilogue.hpp"

