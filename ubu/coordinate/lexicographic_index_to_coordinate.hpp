#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "coordinate.hpp"
#include "detail/compact_row_major_stride.hpp"
#include "detail/index_to_coordinate.hpp"


namespace ubu
{


// XXX ideally, this would be something like coordinate_divide(reverse(coord), reverse(shape))


// precondition:  is_coordinate_into(coord,shape)
// postcondition: is_natural_coordinate_into(result,shape)
template<coordinate C, coordinate S>
  requires congruent<C,S>
constexpr C lexicographic_index_to_coordinate(const C& coord, const S& shape)
{
  return coord;
}


// precondition:  is_coordinate_into(coord,shape)
// postcondition: is_natural_coordinate_into(result,shape)
template<coordinate C, coordinate S>
  requires weakly_congruent<C,S>
constexpr congruent<S> auto lexicographic_index_to_coordinate(const C& coord, const S& shape)
{
  if constexpr(std::integral<C>)
  {
    return detail::index_to_coordinate(coord, shape, detail::compact_row_major_stride(shape));
  }
  else
  {
    return detail::tuple_zip_with(coord, shape, [](auto& c, auto& s)
    {
      return lexicographic_index_to_coordinate(c,s);
    });
  }
}


} // end ubu

#include "../detail/epilogue.hpp"

