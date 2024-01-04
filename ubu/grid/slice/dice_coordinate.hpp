#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "slicer.hpp"
#include "underscore.hpp"
#include <tuple>
#include <utility>

namespace ubu
{
namespace detail
{


// terminal case 1: the slicer is literally the underscore: discard the whole coordinate
template<slicer C>
constexpr std::tuple<> dice_coordinate_impl(const C& coord, detail::underscore_t)
{
  return {};
}

// terminal case 2: the slicer does not contain an underscore; keep the whole coordinate
template<slicer C, slicer_for<C> S>
  requires slicer_without_underscore<S>
constexpr C dice_coordinate_impl(const C& coord, const S&)
{
  return coord;
}

// recursive case: the slicer is nonscalar
// this is a forward declaration for recursive_dice_coordinate_impl
template<slicer C, nonscalar_slicer_for<C> S>
  requires slicer_with_underscore<S>
constexpr auto dice_coordinate_impl(const C& coord, const S& katana);

template<detail::tuple_like R, class Slicer, class Arg>
constexpr auto wrap_if_no_underscore_or_int(const Arg& arg)
{
  if constexpr(slicer_without_underscore<Slicer> or std::integral<Arg>)
  {
    return make_tuple_similar_to<R>(arg);
  }
  else
  {
    return arg;
  }
}

template<slicer C, nonscalar_slicer_for<C> S, std::size_t... I>
constexpr auto recursive_dice_coordinate_impl(std::index_sequence<I...>, const C& coord, const S& katana)
{
  // we apply dice_coordinate to each element of coord & katana and we want to concatenate all the results
  // we also want to preserve the tuple structure of any tuples that were selected by katana at element I
  //
  // so, if element I of katana has no underscore, we wrap that result in an extra tuple layer to preserve the tuple in the concatenation
  // we also need to wrap dice_coordinate's result if it returns a raw integer for tuple_cat to work

  // finally, because it's really inconvenient for this function to return a (single_thing), we unwrap any singles we find

  auto result_tuple = tuple_cat_similar_to<C>(wrap_if_no_underscore_or_int<C,std::tuple_element_t<I,S>>(dice_coordinate_impl(get<I>(coord), get<I>(katana)))...);
  return detail::tuple_unwrap_single(result_tuple);
}


// recursive case: the slicer is nonscalar
template<slicer C, nonscalar_slicer_for<C> S>
  requires slicer_with_underscore<S>
constexpr auto dice_coordinate_impl(const C& coord, const S& katana)
{
  return recursive_dice_coordinate_impl(detail::tuple_indices<S>, coord, katana);
}


} // end detail


template<slicer C, slicer_for<C> S>
constexpr slicer auto dice_coordinate(const C& coord, const S& katana)
{
  return detail::dice_coordinate_impl(coord, katana);
}

} // end ubu

#include "../../detail/epilogue.hpp"

