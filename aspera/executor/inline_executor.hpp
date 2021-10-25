#pragma once

#include "../detail/prologue.hpp"

#include "complete_event.hpp"
#include <compare>
#include <concepts>
#include <functional>


ASPERA_NAMESPACE_OPEN_BRACE


struct inline_executor
{
  auto operator<=>(const inline_executor&) const = default;

  template<std::invocable F>
  complete_event execute(F&& f) const
  {
    std::invoke(std::forward<F>(f));
    return {};
  }
};


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

