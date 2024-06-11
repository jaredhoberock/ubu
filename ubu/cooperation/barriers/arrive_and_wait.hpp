#pragma once

#include "../../detail/prologue.hpp"

#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_arrive_and_wait_member_function = requires(T bar)
{
  bar.arrive_and_wait();
};

template<class T>
concept has_arrive_and_wait_free_function = requires(T bar)
{
  arrive_and_wait(bar);
};

struct dispatch_arrive_and_wait
{
  template<class T>
    requires has_arrive_and_wait_member_function<T&&>
  constexpr void operator()(T&& arg) const
  {
    std::forward<T>(arg).arrive_and_wait();
  }

  template<class T>
    requires (not has_arrive_and_wait_member_function<T&&>
              and has_arrive_and_wait_free_function<T&&>)
  constexpr void operator()(T&& arg) const
  {
    arrive_and_wait(std::forward<T>(arg));
  }
}; // end dispatch_arrive_and_wait


} // end detail


namespace
{

constexpr detail::dispatch_arrive_and_wait arrive_and_wait;

} // end anonymous namespace


} // end ubu

#include "../../detail/epilogue.hpp"

