#pragma once

#include "../detail/prologue.hpp"

#include "../event/event.hpp"
#include "../event/wait.hpp"
#include "executor.hpp"
#include "then_execute.hpp"
#include <concepts>
#include <functional>
#include <future>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class Ev, class F>
concept has_finally_execute_member_function = requires(Ex ex, Ev ev, F f)
{
  ex.finally_execute(ev, f);
};

template<class Ex, class Ev, class F>
concept has_finally_execute_free_function = requires(Ex ex, Ev ev, F f)
{
  finally_execute(ex, ev, f);
};


// this is the type of finally_execute
struct dispatch_finally_execute
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class F>
    requires has_finally_execute_member_function<Ex&&,Ev&&,F&&>
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return std::forward<Ex>(ex).finally_execute(std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class Ex, class Ev, class F>
    requires (!has_finally_execute_member_function<Ex&&,Ev&&,F&&> and has_finally_execute_free_function<Ex&&,Ev&&,F&&>)
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return finally_execute(std::forward<Ex>(ex), std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path adapts then_execute
  template<executor Ex, event Ev, std::invocable F>
    requires (!has_finally_execute_member_function<Ex&&,Ev&&,F&&> and !has_finally_execute_free_function<Ex&&,Ev&&,F&&>)
  void operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    // the event returned by then_execute is discarded
    then_execute(std::forward<Ex>(ex), std::forward<Ev>(ev), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_finally_execute finally_execute;

} // end anonymous namespace


template<class Ex, class Ev, class F>
using finally_execute_result_t = decltype(ASPERA_NAMESPACE::finally_execute(std::declval<Ex>(), std::declval<Ev>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

