#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/concepts/integral_like.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "slicer.hpp"
#include "underscore.hpp"
#include <tuple>
#include <utility>

namespace ubu
{
namespace detail
{


// terminal case 1: katana is literally the underscore: keep the whole coordinate
template<slicer C>
constexpr C slice_coordinate_impl(const C& coord, detail::underscore_t)
{
  return coord;
}

// terminal case 2: katana does not contain an underscore, discard the whole coordinate
template<slicer C, slicer_for<C> K>
  requires slicer_without_underscore<K>
constexpr std::tuple<> slice_coordinate_impl(const C&, const K&)
{
  return {};
}

// recursive case: katana is nonscalar
// this is a forward declaration for recursive_slice_coordinate_impl
template<slicer C, nonscalar_slicer_for<C> K>
  requires slicer_with_underscore<K>
constexpr auto slice_coordinate_impl(const C& coord, const K& katana);

template<detail::tuple_like R, class MaybeUnderscore, class Arg>
constexpr auto wrap_if_underscore_or_int(const Arg& arg)
{
  if constexpr(is_underscore_v<MaybeUnderscore> or integral_like<Arg>)
  {
    return make_tuple_similar_to<R>(arg);
  }
  else
  {
    return arg;
  }
}

template<slicer C, nonscalar_slicer_for<C> K, std::size_t... I>
constexpr auto recursive_slice_coordinate_impl(std::index_sequence<I...>, const C& coord, const K& katana)
{
  // we apply slice_coordinate to each element of coord & katana and we want to concatenate all the results
  // we also want to preserve the tuple structure of any tuples that were selected by underscore at element I
  //
  // so, if element I of katana is an underscore, we wrap that result in an extra tuple layer to preserve the tuple in the concatenation
  // we also need to wrap slice_coordinate's result if it returns a raw integer for tuple_cat to work

  // finally, because it's really inconvenient for this function to return a (single_thing), we unwrap any singles we find

  auto result_tuple = tuple_cat_similar_to<C>(wrap_if_underscore_or_int<C,std::tuple_element_t<I,K>>(slice_coordinate_impl(get<I>(coord), get<I>(katana)))...);
  return detail::tuple_unwrap_single(result_tuple);
}


// recursive case: katana is nonscalar
template<slicer C, nonscalar_slicer_for<C> K>
  requires slicer_with_underscore<K>
constexpr auto slice_coordinate_impl(const C& coord, const K& katana)
{
  return recursive_slice_coordinate_impl(detail::tuple_indices<K>, coord, katana);
}


} // end detail


template<slicer C, slicer_for<C> K>
constexpr slicer auto slice_coordinate(const C& coord, const K& katana)
{
  return detail::slice_coordinate_impl(coord, katana);
}

} // end ubu

#include "../../detail/epilogue.hpp"

