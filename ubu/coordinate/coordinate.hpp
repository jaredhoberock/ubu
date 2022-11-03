#pragma once

#include "../detail/prologue.hpp"

#include "detail/tuple_algorithm.hpp"
#include "element.hpp"
#include "rank.hpp"
#include <concepts>
#include <type_traits>
#include <utility>


namespace ubu::detail
{


template<class T>
struct is_coordinate;


// check T for elements 0... N-1, and make sure that each one is itself a coordinate
template<class T, std::size_t... I>
constexpr bool has_elements_that_are_coordinates(std::index_sequence<I...>)
{
  return (... and is_coordinate<element_t<I,T>>::value);
}


template<class T>
concept rank_one_coordinate =
  static_rank<T>
  and (rank_v<T> == 1)
  and std::integral<element_t<0,T>>
;


template<class T>
concept static_rank_greater_than_one =
  static_rank<T>
  and (rank_v<T> > 1)
;


template<class T>
struct is_coordinate
{
  template<class U = T>
    requires rank_one_coordinate<U>
  static constexpr bool test(int)
  {
    return true;
  }

  template<class U = T>
    requires static_rank_greater_than_one<U>
  static constexpr bool test(int)
  {
    return has_elements_that_are_coordinates<U>(std::make_index_sequence<rank_v<U>>{});
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


// coordinate is a recursive concept, so we need to implement it with traditional SFINAE techniques
template<class T>
concept coordinate = detail::is_coordinate<T>::value;

template<class... Types>
concept are_coordinates = (... and coordinate<Types>);

template<class T, std::size_t N>
concept coordinate_of_rank = coordinate<T> and (rank_v<T> == N);


template<class T>
concept tuple_like_coordinate = coordinate<T> and (rank_v<T> > 1);

template<class... Types>
concept are_tuple_like_coordinates = (... and tuple_like_coordinate<Types>);


} // end ubu


#include "../detail/epilogue.hpp"

