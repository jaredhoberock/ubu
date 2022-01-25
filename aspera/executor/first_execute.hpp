#pragma once

#include "../detail/prologue.hpp"

#include "../event/event.hpp"
#include "execute.hpp"
#include "executor.hpp"
#include <concepts>
#include <functional>
#include <future>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class E, class F>
concept has_first_execute_member_function = requires(E e, F f)
{
  {e.first_execute(f)} -> event;
};

template<class E, class F>
concept has_first_execute_free_function = requires(E e, F f)
{
  {first_execute(e, f)} -> event;
};


// this is the type of first_execute
struct dispatch_first_execute
{
  // this dispatch path calls the member function
  template<class E, class F>
    requires has_first_execute_member_function<E&&,F&&>
  constexpr auto operator()(E&& e, F&& f) const
  {
    return std::forward<E>(e).first_execute(std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class E, class F>
    requires (!has_first_execute_member_function<E&&,F&&> and has_first_execute_free_function<E&&,F&&>)
  constexpr auto operator()(E&& e, F&& f) const
  {
    return first_execute(std::forward<E>(e), std::forward<F>(f));
  }

  // this dispatch path adapts execute
  template<executor E, std::invocable F>
    requires (!has_first_execute_member_function<E&&,F&&> and !has_first_execute_free_function<E&&,F&&>)
  std::future<void> operator()(E&& e, F&& f) const
  {
    std::promise<void> p;
    std::future<void> result = p.get_future();

    execute(std::forward<E>(e), [p=std::move(p), f=std::forward<F>(f)]() mutable
    {
      std::invoke(std::forward<F>(f));
      p.set_value();
    });

    return result;
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_first_execute first_execute;

} // end anonymous namespace


template<class E, class F>
using first_execute_result_t = decltype(ASPERA_NAMESPACE::first_execute(std::declval<E>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

