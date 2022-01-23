#pragma once

#include "../detail/prologue.hpp"

#include "discrete_coordinate.hpp"
#include "element.hpp"
#include "point.hpp"
#include "size.hpp"
#include <concepts>
#include <cstdint>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

// scalar case
template<std::integral I>
constexpr std::size_t size_of_discrete_coordinate_space(const I& shape_of_space)
{
  return static_cast<std::size_t>(shape_of_space);
}


// forward declaration of non-scalar case
template<discrete_coordinate T>
  requires (!std::integral<T>)
constexpr std::size_t size_of_discrete_coordinate_space(const T& shape_of_space);


namespace detail
{


// this function takes an index, calls size_of_index_space on each of its elements,
// and returns a new point<size_t> whose elements are the results
template<discrete_coordinate T, std::size_t... Indices>
constexpr point<std::size_t, size_v<T>> to_tuple_of_sizes(const T& shape_of_space, std::index_sequence<Indices...>)
{
  return {size_of_discrete_coordinate_space(element<Indices>(shape_of_space))...};
}


} // end detail


// non-scalar case
template<discrete_coordinate T>
  requires (!std::integral<T>)
constexpr std::size_t size_of_discrete_coordinate_space(const T& shape_of_space)
{
  // transform shape into a tuple of sizes
  point<std::size_t, size_v<T>> tuple_of_sizes = detail::to_tuple_of_sizes(shape_of_space, std::make_index_sequence<size_v<T>>{});

  // return the product of the sizes
  return tuple_of_sizes.product();
}



ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

