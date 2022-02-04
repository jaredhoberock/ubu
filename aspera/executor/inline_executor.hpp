#pragma once

#include "../detail/prologue.hpp"

#include "../event/complete_event.hpp"
#include "../event/event.hpp"
#include <compare>
#include <concepts>
#include <functional>


ASPERA_NAMESPACE_OPEN_BRACE


struct inline_executor
{
  constexpr bool operator==(const inline_executor&) const
  {
    return true;
  }

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

  template<event E, std::invocable F>
  constexpr complete_event execute_after(E&& before, F&& f) const
  {
    wait(std::forward<E>(before));
    return this->first_execute(std::forward<F>(f));
  }
};


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

