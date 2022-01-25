#pragma once

#include "../detail/prologue.hpp"

#include "../event/complete_event.hpp"
#include <compare>
#include <concepts>
#include <functional>


ASPERA_NAMESPACE_OPEN_BRACE


struct inline_executor
{
  auto operator<=>(const inline_executor&) const = default;

  template<std::invocable F>
  constexpr void execute(F&& f) const
  {
    std::invoke(std::forward<F>(f));
  }

  template<std::invocable F>
  constexpr complete_event first_execute(F&& f) const
  {
    this->execute(std::forward<F>(f));
    return {};
  }
};


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

