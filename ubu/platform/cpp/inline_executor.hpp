#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/causality/actual_happening.hpp"
#include "../../places/causality/past_event.hpp"
#include "../../places/causality/wait.hpp"
#include <concepts>
#include <functional>

namespace ubu
{
inline namespace cpp
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


} // end cpp
} // end ubu

#include "../../detail/epilogue.hpp"

