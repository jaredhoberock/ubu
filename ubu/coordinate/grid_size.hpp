#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "element.hpp"
#include "point.hpp"
#include "rank.hpp"
#include <concepts>
#include <cstdint>
#include <utility>


namespace ubu
{


// scalar case
template<scalar_coordinate C>
constexpr std::size_t grid_size(const C& grid_shape)
{
  return static_cast<std::size_t>(element<0>(grid_shape));
}


// forward declaration of nonscalar case
template<nonscalar_coordinate C>
constexpr std::size_t grid_size(const C& grid_shape);


namespace detail
{


// this function takes a coordinate, calls grid_size on each of its elements,
// and returns a new point<size_t> whose elements are the results
template<nonscalar_coordinate C, std::size_t... Indices>
  requires (sizeof...(Indices) == rank_v<C>)
constexpr point<std::size_t, rank_v<C>> to_tuple_of_sizes(const C& grid_shape, std::index_sequence<Indices...>)
{
  return {grid_size(element<Indices>(grid_shape))...};
}


} // end detail


// nonscalar case
template<nonscalar_coordinate C>
constexpr std::size_t grid_size(const C& grid_shape)
{
  // transform grid_shape into a tuple of sizes
  point<std::size_t, rank_v<C>> tuple_of_sizes = detail::to_tuple_of_sizes(grid_shape, std::make_index_sequence<rank_v<C>>{});

  // return the product of the sizes
  return tuple_of_sizes.product();
}


} // end ubu


#include "../detail/epilogue.hpp"

