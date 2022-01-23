#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include <type_traits>
#include <utility>
#include "coordinate.hpp"
#include "element.hpp"
#include "size.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
struct is_index;


template<std::size_t I, class T>
concept element_is_an_index = (requires(T x) { ASPERA_NAMESPACE::element<I>(x); } and is_index<element_t<I,T>>::value);


// check T for elements 0... N-1, and make sure that each one is itself an index
template<class T, std::size_t... I>
constexpr bool has_elements_that_are_indices(std::index_sequence<I...>)
{
  return (... && element_is_an_index<I,T>);
}


template<class T>
struct is_index
{
  // integral types are indices
  template<class U = T>
    requires std::integral<U>
  static constexpr bool test(int)
  {
    return true;
  }

  // floating point types are not indices
  template<class U = T>
    requires std::floating_point<U>
  static constexpr bool test(int)
  {
    return false;
  }

  // non-scalar coordinates may be indices
  template<class U = T>
    requires (!std::integral<U> and !std::floating_point<U> and coordinate<T>)
  static constexpr bool test(int)
  {
    return has_elements_that_are_indices<U>(std::make_index_sequence<size_v<U>>{});
  }

  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test(0);
};


} // end detail


// index is a recursive concept, so we need to implement it with traditional SFINAE techniques
// it's redundant, but make index a refinement of coordinate for convenience
template<class T>
concept index = (coordinate<T> and detail::is_index<T>::value);


template<class T, std::size_t N>
concept index_of_size = index<T> and (size_v<T> == N);


template<class... Types>
concept are_indices = (... and index<Types>);


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"


