#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "detail/compact_row_major_stride.hpp"
#include "grid_coordinate.hpp"
#include "to_grid_coordinate.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{


template<grid_coordinate S>
constexpr S colexicographic_index_to_grid_coordinate(const std::integral auto& i, const S& shape)
{
  return ubu::to_grid_coordinate<S>(i, shape, detail::compact_row_major_stride(shape));
}


} // end ubu

#include "../detail/epilogue.hpp"

