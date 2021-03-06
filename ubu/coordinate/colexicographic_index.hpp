#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "detail/compact_row_major_stride.hpp"
#include "grid_coordinate.hpp"
#include "to_index.hpp"
#include <cstdint>


namespace ubu
{

template<grid_coordinate C, grid_coordinate S>
  requires congruent<C,S>
constexpr std::size_t colexicographic_index(const C& coord, const S& grid_shape)
{
  return ubu::to_index(coord, grid_shape, detail::compact_row_major_stride(grid_shape));
}

} // end ubu

#include "../detail/epilogue.hpp"

