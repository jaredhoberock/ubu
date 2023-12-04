#pragma once

#include "../../detail/prologue.hpp"

#include <compare>
#include <concepts>
#include <functional>
#include <future>
#include <thread>
#include <utility>


namespace ubu
{
inline namespace cpp
{

struct new_thread_executor
{
  auto operator<=>(const new_thread_executor&) const = default;

  static std::future<void> first_cause()
  {
    std::promise<void> p;
    auto result = p.get_future();
    p.set_value();
    return result;
  }

  template<std::invocable F>
  std::future<void> execute_after(const std::future<void>& before, F&& f) const
  {
    std::promise<void> p;
    auto result = p.get_future();

    std::thread t{[&before, f=std::forward<F>(f), p=std::move(p)]() mutable
    {
      before.wait();
      std::invoke(std::forward<F>(f));
      p.set_value();
    }};

    t.detach();

    return result;
  }

  template<std::invocable F>
  std::future<void> execute_after(std::future<void>&& before, F&& f) const
  {
    std::promise<void> p;
    auto result = p.get_future();

    std::thread t{[before=std::move(before), f=std::forward<F>(f), p=std::move(p)]() mutable
    {
      before.wait();
      std::invoke(std::forward<F>(f));
      p.set_value();
    }};

    t.detach();

    return result;
  }
};


} // end cpp
} // end ubu

#include "../../detail/epilogue.hpp"

