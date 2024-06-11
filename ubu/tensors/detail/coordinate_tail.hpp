#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinates/concepts/semicoordinate.hpp"
#include "../coordinates/traits/rank.hpp"
#include "../coordinates/detail/tuple_algorithm.hpp"

namespace ubu::detail
{

template<semicoordinate C>
  requires (rank_v<C> > 1)
constexpr semicoordinate auto coordinate_tail(const C& coord)
{
  // unwrap any singles so that we return raw integers
  return detail::tuple_unwrap_single(detail::tuple_drop_first(coord));
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

