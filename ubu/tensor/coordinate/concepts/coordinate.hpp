#pragma once

#include "../../../detail/prologue.hpp"

#include "../detail/as_integral.hpp"
#include "../element.hpp"
#include "../traits/rank.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>


namespace ubu
{


// XXX we might need to also insist that zeros<T> exist and its result be std::integral
//     the type of zeros<T> is allowed to differ from T, but it must be comparable to T
template<class T>
concept scalar_coordinate =
  requires(T coord)
  {
    detail::as_integral(coord);
  }
;


namespace detail
{


template<class T>
struct is_nonscalar_coordinate;


// check T for elements 0... N-1, and make sure that each one is itself a coordinate
template<class T, std::size_t... I>
constexpr bool has_elements_that_are_coordinates(std::index_sequence<I...>)
{
  return (... and (scalar_coordinate<std::tuple_element_t<I,T>> or is_nonscalar_coordinate<std::tuple_element_t<I,T>>::value));
}


template<class T>
concept static_rank_greater_than_one =
  static_rank<T>
  and (rank_v<T> > 1)
;


template<class T>
struct is_nonscalar_coordinate
{
  template<class U = T>
    requires static_rank_greater_than_one<U>
  static constexpr bool test(int)
  {
    return has_elements_that_are_coordinates<std::remove_cvref_t<U>>(std::make_index_sequence<rank_v<U>>{});
  }

  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test(0);
};


} // end detail


// nonscalar_coordinate is a recursive concept, so we need to implement it with traditional SFINAE techniques
template<class T>
concept nonscalar_coordinate = detail::is_nonscalar_coordinate<T>::value;


// a coordinate is either a scalar or nonscalar coordinate
template<class T>
concept coordinate = scalar_coordinate<T> or nonscalar_coordinate<T>;

template<class... Types>
concept coordinates = (... and coordinate<Types>);

template<class T, std::size_t N>
concept coordinate_of_rank = coordinate<T> and (rank_v<T> == N);

template<class... Types>
concept nonscalar_coordinates = (... and nonscalar_coordinate<Types>);


// XXX consider reorganizing coordinate_for and element underneath tensor
template<class C, class T>
concept coordinate_for =
  coordinate<C>
  and requires(C coord, T obj)
  {
    ubu::element(obj, coord);
  }
;


} // end ubu


#include "../../../detail/epilogue.hpp"

