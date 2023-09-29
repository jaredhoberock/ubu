#pragma once

#include "../../../detail/prologue.hpp"

#include "../congruent.hpp"
#include "../coordinate.hpp"
#include "../coordinate_cast.hpp"
#include "../coordinate_sum.hpp"
#include "../colexicographical_lift.hpp"
#include <concepts>


namespace ubu
{

template<coordinate C, congruent<C> S, std::integral I>
constexpr void colexicographical_advance(C& coord, const S& shape, I n)
{
  congruent<C> auto delta = colexicographical_lift(n, shape);
  coord = coordinate_cast<C>(coordinate_sum(coord, delta));
}


} // end ubu


#include "../../../detail/epilogue.hpp"

