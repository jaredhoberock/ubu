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

// terminal case 1: the slicer is literally the underscore: keep the whole coordinate
template<coordinate C>
constexpr C slice_coordinate(const C& coord, detail::underscore_t)
{
  return coord;
}

// terminal case 2: the slicer does not contain an underscore, discard the whole coordinate
template<coordinate C, slicer_for<C> S>
  requires slicer_without_underscore<S>
constexpr std::tuple<> slice_coordinate(const C&, const S&)
{
  return {};
}

// recursive case: the slicer is nonscalar
// this is a forward declaration for slice_coordinate_impl
template<coordinate C, nonscalar_slicer_for<C> S>
constexpr auto slice_coordinate(const C& coord, const S& katana);


namespace detail
{

template<coordinate C, nonscalar_slicer_for<C> S, std::size_t... I>
constexpr auto slice_coordinate_impl(std::index_sequence<I...>, const C& coord, const S& katana)
{
  // recursively call slice_coordinate on the elements of tuple, and concatenate the results together
  auto result_tuple = tuple_cat_similar_to<C>(ensure_tuple(slice_coordinate(get<I>(coord), get<I>(katana)))...);

  // unwrap any singles into raw integers
  return tuple_unwrap_single(result_tuple);
}


} // end detail


// recursive case: the slicer is nonscalar
template<coordinate C, nonscalar_slicer_for<C> S>
constexpr auto slice_coordinate(const C& coord, const S& katana)
{
  return detail::slice_coordinate_impl(detail::tuple_indices<S>, coord, katana);
}

} // end ubu

#include "../../detail/epilogue.hpp"

