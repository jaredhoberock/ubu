#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/actual_happening.hpp"
#include "../../causality/past_event.hpp"
#include "../../causality/wait.hpp"
#include <concepts>
#include <functional>


namespace ubu::cpp
{


struct inline_executor
{
  constexpr bool operator==(const inline_executor&) const
  {
    return true;
  }

  using happening_type = past_event;

  template<actual_happening H, std::invocable F>
  constexpr past_event execute_after(H&& before, F&& f) const
  {
    wait(std::forward<H>(before));
    std::invoke(std::forward<F>(f));
    return {};
  }
};


} // end ubu::cpp


#include "../../detail/epilogue.hpp"

