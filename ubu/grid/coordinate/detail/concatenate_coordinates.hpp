#pragma once

#include "../../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "../point.hpp"
#include "tuple_algorithm.hpp"

namespace ubu::detail
{


template<coordinate C>
constexpr C concatenate_coordinates(const C& coord)
{
  return coord;
}


template<coordinate C, coordinate... Cs>
constexpr coordinate auto concatenate_coordinates(const C& coord, const Cs&... coords)
{
  return tuple_cat(ensure_tuple_similar_to<int2>(coord), ensure_tuple_similar_to<int2>(coords)...);
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

