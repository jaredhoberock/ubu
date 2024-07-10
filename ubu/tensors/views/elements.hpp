#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "../concepts/tensor_like.hpp"
#include "../traits/tensor_element.hpp"
#include "transform.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu
{

template<class T>
concept tensor_like_of_tuple_like =
  tensor_like<T> and
  tuples::tuple_like<tensor_reference_t<T>>
;

template<class T, std::size_t I>
concept tensor_like_with_elements =
  // T must be a tensor of tuples
  tensor_like_of_tuple_like<T> and
  // I must be a valid index into the tuple
  I < std::tuple_size_v<std::remove_cvref_t<tensor_reference_t<T>>> and
  // either T[coord] is a reference (to a tuple) or tuple[I] is a reference
  (std::is_reference_v<tensor_reference_t<T>> or 
   std::is_reference_v<std::tuple_element_t<I, std::remove_cvref_t<tensor_reference_t<T>>>>)
;

template<std::size_t I, tensor_like_with_elements<I> T>
constexpr view auto elements(T&& tensor)
{
  return transform(std::forward<T>(tensor), [](auto&& tuple) -> decltype(auto)
  {
    return get<I>(std::forward<decltype(tuple)>(tuple));
  });
}

} // end ubu

#include "../../detail/epilogue.hpp"

