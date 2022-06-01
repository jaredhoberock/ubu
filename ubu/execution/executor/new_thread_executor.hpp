#pragma once

#include "../../detail/prologue.hpp"

#include <compare>
#include <concepts>
#include <thread>
#include <utility>


namespace ubu
{


struct new_thread_executor
{
  auto operator<=>(const new_thread_executor&) const = default;

  template<std::invocable F>
  void execute(F&& f) const
  {
    std::thread t{std::forward<F>(f)};
    t.detach();
  }
};


} // end ubu


#include "../../detail/epilogue.hpp"

