#pragma once

#include "../detail/prologue.hpp"

#include "element.hpp"
#include "grid_coordinate.hpp"
#include "point.hpp"
#include "rank.hpp"
#include <concepts>
#include <cstdint>
#include <utility>


namespace ubu
{


// scalar case
constexpr std::size_t grid_size(const std::integral auto& grid_shape)
{
  return static_cast<std::size_t>(grid_shape);
}


// forward declaration of non-scalar case
template<grid_coordinate T>
  requires (!std::integral<T>)
constexpr std::size_t grid_size(const T& grid_shape);


namespace detail
{


// this function takes a grid_coordinate, calls grid_size on each of its elements,
// and returns a new point<size_t> whose elements are the results
template<grid_coordinate T, std::size_t... Indices>
constexpr point<std::size_t, rank_v<T>> to_tuple_of_sizes(const T& grid_shape, std::index_sequence<Indices...>)
{
  return {grid_size(element<Indices>(grid_shape))...};
}


} // end detail


// non-scalar case
template<grid_coordinate T>
  requires (!std::integral<T>)
constexpr std::size_t grid_size(const T& grid_shape)
{
  // transform grid_shape into a tuple of sizes
  point<std::size_t, rank_v<T>> tuple_of_sizes = detail::to_tuple_of_sizes(grid_shape, std::make_index_sequence<rank_v<T>>{});

  // return the product of the sizes
  return tuple_of_sizes.product();
}


} // end ubu


#include "../detail/epilogue.hpp"

