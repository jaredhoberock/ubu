#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../event/wait.hpp"
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
concept has_execute_after_member_function = requires(Ex ex, Ev ev, F f)
{
  {ex.execute_after(ev, f)} -> event;
};

template<class Ex, class Ev, class F>
concept has_execute_after_free_function = requires(Ex ex, Ev ev, F f)
{
  {execute_after(ex, ev, f)} -> event;
};


// this is the type of execute_after
struct dispatch_execute_after
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class F>
    requires has_execute_after_member_function<Ex&&,Ev&&,F&&>
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return std::forward<Ex>(ex).execute_after(std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class Ex, class Ev, class F>
    requires (!has_execute_after_member_function<Ex&&,Ev&&,F&&> and has_execute_after_free_function<Ex&&,Ev&&,F&&>)
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return execute_after(std::forward<Ex>(ex), std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path adapts first_execute
  template<executor Ex, event Ev, std::invocable F>
    requires (!has_execute_after_member_function<Ex&&,Ev&&,F&&> and !has_execute_after_free_function<Ex&&,Ev&&,F&&>)
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

constexpr detail::dispatch_execute_after execute_after;

} // end anonymous namespace


template<class Ex, class Ev, class F>
using execute_after_result_t = decltype(ASPERA_NAMESPACE::execute_after(std::declval<Ex>(), std::declval<Ev>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

