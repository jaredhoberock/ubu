#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../coordinate_cast.hpp"
#include "../coordinate_sum.hpp"
#include "../lexicographical_lift.hpp"
#include <concepts>


namespace ubu
{

template<coordinate C, congruent<C> S, std::integral I>
constexpr void lexicographical_advance(C& coord, const S& shape, I n)
{
  congruent<C> auto delta = lexicographical_lift(n, shape);
  coord = coordinate_cast<C>(coordinate_sum(coord, delta));
}


} // end ubu


#include "../../../detail/epilogue.hpp"

