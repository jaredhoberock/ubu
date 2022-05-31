#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "detail/compact_row_major_stride.hpp"
#include "grid_coordinate.hpp"
#include "to_index.hpp"
#include <cstdint>


UBU_NAMESPACE_OPEN_BRACE

template<grid_coordinate C, grid_coordinate S>
  requires congruent<C,S>
constexpr std::size_t colexicographic_index(const C& coord, const S& grid_shape)
{
  return UBU_NAMESPACE::to_index(coord, grid_shape, detail::compact_row_major_stride(grid_shape));
}


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

