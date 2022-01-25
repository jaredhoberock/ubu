#pragma once

#include "../detail/prologue.hpp"

#include "../event/event.hpp"
#include "../event/wait.hpp"
#include "executor.hpp"
#include "executor_event.hpp"
#include "first_execute.hpp"
#include <concepts>
#include <functional>
#include <future>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class Ev, class F>
concept has_then_execute_member_function = requires(Ex ex, Ev ev, F f)
{
  {ex.then_execute(ev, f)} -> event;
};

template<class Ex, class Ev, class F>
concept has_then_execute_free_function = requires(Ex ex, Ev ev, F f)
{
  {then_execute(ex, ev, f)} -> event;
};


// this is the type of then_execute
struct dispatch_then_execute
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class F>
    requires has_then_execute_member_function<Ex&&,Ev&&,F&&>
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return std::forward<Ex>(ex).then_execute(std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class Ex, class Ev, class F>
    requires (!has_then_execute_member_function<Ex&&,Ev&&,F&&> and has_then_execute_free_function<Ex&&,Ev&&,F&&>)
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return then_execute(std::forward<Ex>(ex), std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path adapts first_execute
  template<executor Ex, event Ev, std::invocable F>
    requires (!has_then_execute_member_function<Ex&&,Ev&&,F&&> and !has_then_execute_free_function<Ex&&,Ev&&,F&&>)
  executor_event_t<Ex&&> operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return first_execute(std::forward<Ex>(ex), [ev=std::move(ev), f=std::forward<F>(f)]() mutable
    {
      ASPERA_NAMESPACE::wait(std::move(ev));
      std::invoke(std::forward<F>(f));
    });
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_then_execute then_execute;

} // end anonymous namespace


template<class Ex, class Ev, class F>
using then_execute_result_t = decltype(ASPERA_NAMESPACE::then_execute(std::declval<Ex>(), std::declval<Ev>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

