#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor_like.hpp"
#include "../coordinates/detail/tuple_algorithm.hpp"
#include "../traits/tensor_element.hpp"
#include "elements.hpp"
#include <type_traits>
#include <tuple>
#include <utility>

namespace ubu
{
namespace detail
{

template<tuple_like T, std::size_t... Is>
constexpr bool tuple_elements_are_references(std::index_sequence<Is...>)
{
  return (... and std::is_reference_v<std::tuple_element_t<Is,T>>);
}

template<class T>
concept tuple_like_of_references =
  tuple_like<T> and
  tuple_elements_are_references<T>(tuple_indices<T>)
;

} // end detail

template<tensor_like T>
  requires detail::tuple_like_of_references<tensor_reference_t<T>>
constexpr detail::tuple_like auto unzip(T&& tensor_of_tuples)
{
  constexpr std::size_t N = std::tuple_size_v<tensor_reference_t<T>>;

  return [&]<std::size_t... Is>(std::index_sequence<Is...>)
  {
    return std::forward_as_tuple(elements<Is>(tensor_of_tuples)...);
  }(std::make_index_sequence<N>());
}

} // end ubu

#include "../../detail/epilogue.hpp"

