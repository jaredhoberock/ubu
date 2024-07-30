#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/tuples.hpp"
#include "concepts/semicoordinate.hpp"
#include "traits/rank.hpp"
#include <utility>

namespace ubu
{

template<std::size_t N, semicoordinate C>
  requires (N < rank_v<C>)
constexpr tuples::pair_like auto split_coordinate_at(const C& coord)
{
  auto [left, right] = tuples::split_at<N>(tuples::ensure_tuple(coord));

  return std::pair(tuples::unwrap_single(left), tuples::unwrap_single(right));
}

template<std::size_t N, semicoordinate C>
  requires (N < rank_v<C>)
using split_coordinate_at_t = decltype(split_coordinate_at<N>(std::declval<C>()));

} // end ubu

#include "../../detail/epilogue.hpp"

