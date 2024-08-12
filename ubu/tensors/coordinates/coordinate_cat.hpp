#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "concepts/semicoordinate.hpp"
#include <utility>

namespace ubu
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

template<semicoordinate C1, semicoordinate C2>
using coordinate_cat_result_t = decltype(coordinate_cat(std::declval<C1>(), std::declval<C2>()));

} // end ubu

#include "../../detail/epilogue.hpp"

