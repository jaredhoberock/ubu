#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor_like.hpp"
#include "../traits/tensor_element.hpp"
#include "all.hpp"
#include "transform.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

template<std::size_t I, tensor_like T>
  requires (detail::tuple_like<tensor_reference_t<T>> and
            I < std::tuple_size_v<tensor_reference_t<T>> and
            std::is_reference_v<std::tuple_element_t<I,tensor_reference_t<T>>>)
constexpr view auto elements(T&& tensor)
{
  return transform(std::forward<T>(tensor), [](T&& element) -> decltype(auto)
  {
    return get<I>(std::forward<decltype(element)>(element));
  });
}

} // end ubu

#include "../../detail/epilogue.hpp"

