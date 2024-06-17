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

template<tensor_like T, std::size_t... Is>
constexpr bool unzippable_tensor_like_impl(std::index_sequence<Is...>)
{
  return (... and tensor_like_with_elements<T,Is>);
}

} // end detail

template<class T>
concept unzippable_tensor_like = 
  tensor_like_of_tuple_like<T> and
  detail::unzippable_tensor_like_impl<T>(detail::tuple_indices<tensor_reference_t<T>>)
;

template<unzippable_tensor_like T>
constexpr detail::tuple_like auto unzip(T&& tensor_of_tuples)
{
  constexpr std::size_t N = std::tuple_size_v<std::remove_cvref_t<tensor_element_t<T>>>;

  return [&]<std::size_t... Is>(std::index_sequence<Is...>)
  {
    return std::tuple(elements<Is>(tensor_of_tuples)...);
  }(std::make_index_sequence<N>());
}

} // end ubu

#include "../../detail/epilogue.hpp"

