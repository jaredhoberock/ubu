#pragma once

#include "../detail/prologue.hpp"

#include "all.hpp"
#include "concepts/tensor_like.hpp"
#include "iterator.hpp"
#include <ranges>
#include <utility>

namespace ubu
{

template<tensor_like T>
constexpr std::ranges::range auto enumerate(T&& tensor)
{
  auto view = all(std::forward<T>(tensor));

  enumerated_tensor_iterator begin(view);
  tensor_sentinel end;

  return std::ranges::subrange(begin, end);
}

} // end ubu

#include "../detail/epilogue.hpp"

