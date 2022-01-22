#pragma once

#include "../detail/prologue.hpp"

#include <type_traits>
#include <utility>
#include "detail/number.hpp"
#include "element.hpp"
#include "size.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
struct is_coordinate;


template<std::size_t I, class T>
concept element_is_a_coordinate = (requires(T x) { ASPERA_NAMESPACE::element<I>(x); } and is_coordinate<element_t<I,T>>::value);


// check T for elements 0... N-1, and make sure that each one is itself a coordinate
template<class T, std::size_t... I>
constexpr bool has_elements_that_are_coordinates(std::index_sequence<I...>)
{
  return (... && element_is_a_coordinate<I,T>);
}


template<class T>
struct is_coordinate
{
  template<class U = T>
    requires detail::number<U>
  static constexpr bool test(int)
  {
    return true;
  }

  template<class U = T>
    requires (!detail::number<U> and detail::has_static_size<U>)
  static constexpr bool test(int)
  {
    return has_elements_that_are_coordinates<U>(std::make_index_sequence<size_v<U>>{});
  }

  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test(0);
};


} // end detail


// coordinate is a recursive concept, so we need to implement it with traditional SFINAE techniques
template<class T>
concept coordinate = detail::is_coordinate<T>::value;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

