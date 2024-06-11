#pragma once

#include <concepts>
#include <functional>
#include <optional>

template<class T, std::invocable<T,T> F>
  requires std::convertible_to<T, std::invoke_result_t<F,T,T>>
constexpr std::optional<T> maybe_add(std::optional<T> lhs, std::optional<T> rhs, F op)
{
  if(lhs)
  {
    if(rhs)
    {
      return std::invoke(op, *lhs, *rhs);
    }
    else
    {
      return lhs;
    }
  }

  return rhs;
}

