#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include <type_traits>
#include <utility>
#include "coordinate.hpp"
#include "element.hpp"
#include "rank.hpp"


namespace ubu::detail
{


template<class T>
struct is_grid_coordinate;


template<std::size_t I, class T>
concept element_is_a_grid_coordinate = (requires(T x) { ubu::element<I>(x); } and is_grid_coordinate<element_t<I,T>>::value);


// check T for elements 0... N-1, and make sure that each one is itself a grid coordinate
template<class T, std::size_t... I>
constexpr bool has_elements_that_are_grid_coordinates(std::index_sequence<I...>)
{
  return (... && element_is_a_grid_coordinate<I,T>);
}


template<class T>
struct is_grid_coordinate
{
  // integral types are grid coordinates
  template<class U = T>
    requires std::integral<U>
  static constexpr bool test(int)
  {
    return true;
  }

  // floating point types are not grid coordinates
  template<class U = T>
    requires std::floating_point<U>
  static constexpr bool test(int)
  {
    return false;
  }

  // non-scalar coordinates may be grid coordinates
  template<class U = T>
    requires (!std::integral<U> and !std::floating_point<U> and coordinate<T>)
  static constexpr bool test(int)
  {
    return has_elements_that_are_grid_coordinates<U>(std::make_index_sequence<rank_v<U>>{});
  }

  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test(0);
};


} // end ubu::detail


namespace ubu
{


// grid_coordinate is a recursive concept, so we need to implement it with traditional SFINAE techniques
// it's redundant, but make index a refinement of coordinate for convenience
template<class T>
concept grid_coordinate = (coordinate<T> and detail::is_grid_coordinate<T>::value);


template<class T, std::size_t N>
concept grid_coordinate_of_rank = grid_coordinate<T> and (rank_v<T> == N);


template<class... Types>
concept are_grid_coordinates = (... and grid_coordinate<Types>);


template<class T>
concept tuple_like_grid_coordinate = (grid_coordinate<T> and not std::integral<T>);


template<class... Types>
concept are_tuple_like_grid_coordinates = (... and tuple_like_grid_coordinate<Types>);


} // end ubu


#include "../detail/epilogue.hpp"

