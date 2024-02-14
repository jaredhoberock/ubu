#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "../slice/slicer.hpp"

namespace ubu::detail
{

// XXX this function ought to be variadic
template<slicer C1, coordinate C2>
constexpr auto coordinate_cat(const C1& coord1, const C2& coord2)
{
  // ensure that both coordinates are tuples before calling tuple_cat
  auto tuple1 = ensure_tuple(coord1);
  auto tuple2 = ensure_tuple(coord2);

  return tuple_cat(tuple1,tuple2);
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

