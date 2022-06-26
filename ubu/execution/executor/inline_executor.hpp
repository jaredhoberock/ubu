#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../event/past_event.hpp"
#include <compare>
#include <concepts>
#include <functional>


namespace ubu
{


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
  constexpr past_event first_execute(F&& f) const
  {
    this->execute(std::forward<F>(f));
    return {};
  }

  template<event E, std::invocable F>
  constexpr past_event execute_after(E&& before, F&& f) const
  {
    wait(std::forward<E>(before));
    return this->first_execute(std::forward<F>(f));
  }
};


} // end ubu


#include "../../detail/epilogue.hpp"

