#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/coordinate.hpp"
#include "../coordinate/detail/tuple_algorithm.hpp"
#include "slicer.hpp"
#include "underscore.hpp"
#include <tuple>
#include <utility>

namespace ubu
{

// unlike slice_coordinate, dice_coordinate does not require its first parameter to be a coordinate
// in order to allow a slicer to dice itself

// terminal case 1: the slicer is literally the underscore; discard the whole coordinate
template<class C>
constexpr std::tuple<> dice_coordinate(const C&, detail::underscore_t)
{
  return {};
}

// terminal case 2: the slicer does not contain an underscore; keep the whole coordinate
template<class C, slicer_for<C> S>
  requires slicer_without_underscore<S>
constexpr C dice_coordinate(const C coord, const S&)
{
  return coord;
}

// recursive case: the slicer is nonscalar
// this is a forward declaration for dice_coordinate_impl
template<detail::tuple_like C, nonscalar_slicer_for<C> S>
constexpr auto dice_coordinate(const C& coord, const S& katana);


namespace detail
{

template<tuple_like C, nonscalar_slicer_for<C> S, std::size_t... I>
constexpr auto dice_coordinate_impl(std::index_sequence<I...>, const C& coord, const S& katana)
{
  // recursively call dice_coordinate on the elements of tuple, and concatenate the results together
  auto result_tuple = tuple_cat_similar_to<C>(ensure_tuple(dice_coordinate(get<I>(coord), get<I>(katana)))...);

  // unwrap any singles into raw integers
  return tuple_unwrap_single(result_tuple);
}

} // end detail


// recursive case: the slicer is nonscalar
template<detail::tuple_like C, nonscalar_slicer_for<C> S>
constexpr auto dice_coordinate(const C& coord, const S& katana)
{
  return detail::dice_coordinate_impl(detail::tuple_indices<S>, coord, katana);
}

} // end ubu

#include "../../detail/epilogue.hpp"

