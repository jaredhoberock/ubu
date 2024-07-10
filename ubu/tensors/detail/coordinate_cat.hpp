#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "../coordinates/concepts/semicoordinate.hpp"

namespace ubu::detail
{

// XXX this function ought to be variadic
template<semicoordinate C1, semicoordinate C2>
constexpr semicoordinate auto coordinate_cat(const C1& coord1, const C2& coord2)
{
  // ensure that both coordinates are tuples before calling concatenate
  auto tuple1 = tuples::ensure_tuple(coord1);
  auto tuple2 = tuples::ensure_tuple(coord2);

  return tuples::concatenate_like<decltype(tuple1)>(tuple1,tuple2);
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

