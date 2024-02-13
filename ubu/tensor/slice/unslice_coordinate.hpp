#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "slicer.hpp"
#include "underscore.hpp"
#include "unslicer.hpp"
#include <tuple>
#include <utility>

namespace ubu
{
namespace detail
{


template<slicer K, class... Cs>
constexpr slicer auto unslice_coordinate_impl(const K& katana, const Cs&... cs)
{
  if constexpr(is_underscore_v<K>)
  {
    // terminal case 1: katana is literally the underscore;
    // return the first c (there must be a single c)
    static_assert(sizeof...(Cs) == 1);

    return get<0>(std::make_tuple(cs...));
  }
  else if constexpr(slicer_without_underscore<K>)
  {
    // terminal case 2: katana contains no underscore: return katana
    return katana;
  }
  else
  {
    // recursive case: katana must be a tuple
    static_assert(tuple_like<K>);

    // separate the katana into its head and tail
    auto katana_head = get<0>(katana);
    tuple_like auto katana_tail = tuple_drop_first(katana);

    // count the number of underscores in katana's head
    constexpr std::size_t n = underscore_count_v<decltype(katana_head)>;

    // tuple the cs
    tuple_like auto tupled_cs = std::make_tuple(cs...);

    // separate the cs into the first n going towards the katana's head
    // and the remainder going to the tail
    tuple_like auto head_cs = tuple_take<n>(tupled_cs);
    tuple_like auto tail_cs = tuple_drop<n>(tupled_cs);

    // recurse on the katana's head
    auto result_head = unpack_and_invoke(head_cs, [&](const auto&... cs)
    {
      return unslice_coordinate_impl(katana_head, cs...);
    });

    // recurse on the katana's tail
    auto result_tail = unpack_and_invoke(tail_cs, [&](const auto&... cs)
    {
      return unslice_coordinate_impl(katana_tail, cs...);
    });

    // reconstruct the substituted tuple
    return tuple_prepend_similar_to<K>(result_tail, result_head);
  }
}


} // end detail


template<ubu::slicer C, ubu::unslicer_for<C> K>
constexpr ubu::slicer auto unslice_coordinate(const C& coord, const K& katana)
{
  // we may need to unpack coord into its constituent elements when calling the impl

  if constexpr(detail::is_underscore_v<K>)
  {
    // katana is literally the underscore, just return coord
    return coord;
  }
  else if constexpr(coordinate<C> and rank_v<C> == 1)
  {
    // we don't unpack rank-1 coordinates (even if they are singles) when calling the impl
    return detail::unslice_coordinate_impl(katana, coord);
  }
  else if constexpr(detail::tuple_like<C> and std::tuple_size_v<C> == 1)
  {
    // we don't unpack singles when calling the impl
    // an example of a single that is not a coordinate would be (_)
    return detail::unslice_coordinate_impl(katana, coord);
  }
  else
  {
    // if coord's rank is not 1 then it must be a tuple
    // either it is empty or rank > 1
    static_assert(detail::tuple_like<C>);

    // unpack the tuple and call the implementation
    return detail::unpack_and_invoke(coord, [&](const auto&... cs)
    {
      return detail::unslice_coordinate_impl(katana, cs...);
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

