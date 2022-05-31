#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../event/wait.hpp"
#include "executor.hpp"
#include "execute_after.hpp"
#include <concepts>
#include <functional>
#include <future>
#include <utility>


UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class Ev, class F>
concept has_finally_execute_after_member_function = requires(Ex ex, Ev ev, F f)
{
  ex.finally_execute_after(ev, f);
};

template<class Ex, class Ev, class F>
concept has_finally_execute_after_free_function = requires(Ex ex, Ev ev, F f)
{
  finally_execute_after(ex, ev, f);
};


// this is the type of finally_execute_after
struct dispatch_finally_execute_after
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class F>
    requires has_finally_execute_after_member_function<Ex&&,Ev&&,F&&>
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return std::forward<Ex>(ex).finally_execute_after(std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class Ex, class Ev, class F>
    requires (!has_finally_execute_after_member_function<Ex&&,Ev&&,F&&> and has_finally_execute_after_free_function<Ex&&,Ev&&,F&&>)
  constexpr auto operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    return finally_execute_after(std::forward<Ex>(ex), std::forward<Ev>(ev), std::forward<F>(f));
  }

  // this dispatch path adapts execute_after
  template<executor Ex, event Ev, std::invocable F>
    requires (!has_finally_execute_after_member_function<Ex&&,Ev&&,F&&> and !has_finally_execute_after_free_function<Ex&&,Ev&&,F&&>)
  void operator()(Ex&& ex, Ev&& ev, F&& f) const
  {
    // the event returned by execute_after is discarded
    execute_after(std::forward<Ex>(ex), std::forward<Ev>(ev), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_finally_execute_after finally_execute_after;

} // end anonymous namespace


template<class Ex, class Ev, class F>
using finally_execute_after_result_t = decltype(UBU_NAMESPACE::finally_execute_after(std::declval<Ex>(), std::declval<Ev>(), std::declval<F>()));


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

