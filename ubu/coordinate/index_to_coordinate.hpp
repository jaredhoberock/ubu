#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "coordinate.hpp"
#include "lift_coordinate.hpp"


namespace ubu
{


// precondition:  is_coordinate_into(coord,shape)
// postcondition: is_natural_coordinate_into(result,shape)
template<coordinate C, coordinate S>
  requires weakly_congruent<C,S>
constexpr congruent<S> auto index_to_coordinate(const C& coord, const S& shape)
{
  return lift_coordinate(coord, shape);
}


} // end ubu

#include "../detail/epilogue.hpp"

