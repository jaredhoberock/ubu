#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "detail/compact_row_major_stride.hpp"
#include "detail/index_to_coordinate.hpp"
#include "detail/tuple_algorithm.hpp"
#include "grid_coordinate.hpp"
#include "weakly_congruent.hpp"
#include <concepts>


namespace ubu
{


// precondition:  is_coordinate_into(coord,shape)
// postcondition: is_natural_coordinate_into(result,shape)
template<grid_coordinate C, grid_coordinate S>
  requires congruent<C,S>
constexpr C colexicographic_index_to_coordinate(const C& coord, const S& shape)
{
  return coord;
}


// precondition:  is_coordinate_into(coord,shape)
// postcondition: is_natural_coordinate_into(result,shape)
template<grid_coordinate C, grid_coordinate S>
  requires weakly_congruent<C,S>
constexpr congruent<S> auto colexicographic_index_to_coordinate(const C& coord, const S& shape)
{
  if constexpr(std::integral<C>)
  {
    return detail::index_to_coordinate(coord, shape, detail::compact_row_major_stride(shape));
  }
  else
  {
    return detail::tuple_zip_with([](auto& c, auto& s)
    {
      return colexicographic_index_to_coordinate(c,s);
    }, coord, shape);
  }
}


} // end ubu

#include "../detail/epilogue.hpp"

