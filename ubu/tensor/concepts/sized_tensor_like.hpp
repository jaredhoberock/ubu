#pragma once

#include "../../detail/prologue.hpp"
#include "tensor_like.hpp"
#include <ranges>

namespace ubu
{

template<class T>
concept sized_tensor_like =
  tensor_like<T>
  and requires(T t)
  {
    std::ranges::size(t);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

