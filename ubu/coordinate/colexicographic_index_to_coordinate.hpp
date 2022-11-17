#pragma once

#include "../detail/prologue.hpp"

#include "colexicographic_lift.hpp"
#include "congruent.hpp"
#include "coordinate.hpp"


namespace ubu
{


// precondition:  is_coordinate_into(coord,shape)
// postcondition: is_natural_coordinate_into(result,shape)
template<coordinate C, coordinate S>
  requires weakly_congruent<C,S>
constexpr congruent<S> auto colexicographic_index_to_coordinate(const C& coord, const S& shape)
{
  auto [_, remainder] = colexicographic_lift(coord, shape);
  return remainder;
}


} // end ubu

#include "../detail/epilogue.hpp"

