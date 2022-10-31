#pragma once

#include "../detail/prologue.hpp"

#include "grid_coordinate.hpp"
#include "lexicographic_index_to_coordinate.hpp"
#include "weakly_congruent.hpp"
#include <concepts>


namespace ubu
{


// to_natural_coordinate is a synonym for lexicographic_index_to_coordinate
template<ubu::grid_coordinate C, ubu::grid_coordinate S>
  requires ubu::weakly_congruent<C,S>
constexpr ubu::congruent<S> auto to_natural_coordinate(const C& coord, const S& shape)
{
  return lexicographic_index_to_coordinate(coord, shape);
}


} // end ubu

#include "../detail/epilogue.hpp"

