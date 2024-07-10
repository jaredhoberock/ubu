#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "colexicographical_lift.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"


namespace ubu
{
namespace detail
{

template<scalar_coordinate C>
constexpr C deep_reverse(const C& coord)
{
  return coord;
}

template<nonscalar_coordinate C>
constexpr nonscalar_coordinate auto deep_reverse(const C& coord)
{
  // recursively reverse coord
  return tuples::zip_with(tuples::reverse(coord), [](const auto& e)
  {
    return deep_reverse(e);
  });
}

} // end detail

// lexicographical_lift "upcasts" a weakly_congruent coordinate into a higher-dimensional space described by a shape
// because this lift operation is lexicographical, it "aligns" the modes of coord and shape at the right, and proceeds from right to left
// when the coordinate is congruent with the shape, lexicographical_lift is the identity function

template<coordinate C1, coordinate C2>
  requires weakly_congruent<C1,C2>
constexpr congruent<C2> auto lexicographical_lift(const C1& coord, const C2& shape)
{
  // we will implement this function using colexicographical_lift
  // to do so, we'll "deeply" reverse coord and shape
  // and then return the "deep" reverse of the result
  auto reversed_coord = detail::deep_reverse(coord);
  auto reversed_shape = detail::deep_reverse(shape);
  auto reversed_result = colexicographical_lift(reversed_coord, reversed_shape);
  return detail::deep_reverse(reversed_result);
}


} // end ubu

#include "../../detail/epilogue.hpp"

