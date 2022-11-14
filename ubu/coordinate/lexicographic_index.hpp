#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "coordinate.hpp"
#include "detail/compact_column_major_stride.hpp"
#include "to_index.hpp"
#include <cstdint>


namespace ubu
{

template<coordinate C, coordinate S>
  requires congruent<C,S>
constexpr std::size_t lexicographic_index(const C& coord, const S& grid_shape)
{
  return ubu::to_index(coord, grid_shape, detail::compact_column_major_stride(grid_shape));
}

} // end ubu

#include "../detail/epilogue.hpp"

