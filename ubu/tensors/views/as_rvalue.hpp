#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../traits/tensor_reference.hpp"
#include "transform.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{


template<tensor_like T>
  requires std::is_reference_v<tensor_reference_t<T>>
constexpr view auto as_rvalue(T&& tensor)
{
  return transform(std::forward<T>(tensor), [](auto&& element) -> decltype(auto)
  {
    return std::move(element);
  });
}


} // end ubu

#include "../../detail/epilogue.hpp"

