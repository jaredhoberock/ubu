#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/tuples.hpp"
#include "../../coordinates/concepts/coordinate.hpp"
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
    namespace t = tuples;

    // recursive case: katana must be a tuple
    static_assert(t::tuple_like<K>);

    // separate the katana into its head and tail
    auto katana_head = get<0>(katana);
    t::tuple_like auto katana_tail = t::drop_first(katana);

    // count the number of underscores in katana's head
    constexpr std::size_t n = underscore_count_v<decltype(katana_head)>;

    // tuple the cs
    t::tuple_like auto tupled_cs = std::make_tuple(cs...);

    // separate the cs into the first n going towards the katana's head
    // and the remainder going to the tail
    t::tuple_like auto head_cs = t::take<n>(tupled_cs);
    t::tuple_like auto tail_cs = t::drop<n>(tupled_cs);

    // recurse on the katana's head
    auto result_head = t::unpack_and_invoke(head_cs, [&](const auto&... cs)
    {
      return unslice_coordinate_impl(katana_head, cs...);
    });

    // recurse on the katana's tail
    auto result_tail = t::unpack_and_invoke(tail_cs, [&](const auto&... cs)
    {
      return unslice_coordinate_impl(katana_tail, cs...);
    });

    // reconstruct the substituted tuple
    return t::prepend_similar_to<K>(result_tail, result_head);
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
  else if constexpr(tuples::tuple_like<C> and tuples::size_v<C> == 1)
  {
    // we don't unpack singles when calling the impl
    // an example of a single that is not a coordinate would be (_)
    return detail::unslice_coordinate_impl(katana, coord);
  }
  else
  {
    // if coord's rank is not 1 then it must be a tuple
    // either it is empty or rank > 1
    static_assert(tuples::tuple_like<C>);

    // unpack the tuple and call the implementation
    return tuples::unpack_and_invoke(coord, [&](const auto&... cs)
    {
      return detail::unslice_coordinate_impl(katana, cs...);
    });
  }
}


} // end ubu

#include "../../../detail/epilogue.hpp"

