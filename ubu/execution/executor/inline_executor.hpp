#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/actual_happening.hpp"
#include "../../causality/past_event.hpp"
#include "../../causality/wait.hpp"
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

  template<actual_happening H, std::invocable F>
  constexpr past_event execute_after(H&& before, F&& f) const
  {
    wait(std::forward<H>(before));
    return this->first_execute(std::forward<F>(f));
  }
};


} // end ubu


#include "../../detail/epilogue.hpp"

