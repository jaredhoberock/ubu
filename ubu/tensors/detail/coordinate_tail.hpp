#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/tuples.hpp"
#include "../coordinates/concepts/semicoordinate.hpp"
#include "../coordinates/traits/rank.hpp"

namespace ubu::detail
{

template<semicoordinate C>
  requires (rank_v<C> > 1)
constexpr semicoordinate auto coordinate_tail(const C& coord)
{
  // unwrap any singles so that we return raw integers
  return tuples::drop_first_and_unwrap_single(coord);
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

