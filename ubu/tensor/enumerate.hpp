#pragma once

#include <ranges>
#include <ubu/ubu.hpp>
#include <utility>

template<ubu::tensor_like T>
constexpr std::ranges::range auto enumerate(T&& tensor)
{
  auto view = ubu::all(std::forward<T>(tensor));

  ubu::enumerated_tensor_iterator begin(view);
  ubu::tensor_sentinel end;

  return std::ranges::subrange(begin, end);
}

