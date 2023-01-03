#pragma once

#include "../../detail/prologue.hpp"

#include "congruent.hpp"
#include "coordinate.hpp"
#include "../layout/stride/compact_stride.hpp"
#include "to_index.hpp"


namespace ubu
{

template<coordinate C, coordinate S>
  requires congruent<C,S>
constexpr std::size_t coordinate_to_index(const C& coord, const S& grid_shape)
{
  return to_index(coord, grid_shape, compact_stride(grid_shape));
}

} // end ubu

#include "../../detail/epilogue.hpp"

