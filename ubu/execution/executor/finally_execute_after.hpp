#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "concepts/executor.hpp"
#include "execute_after.hpp"
#include <concepts>
#include <functional>
#include <future>
#include <utility>


namespace ubu
{

namespace detail
{


template<class E, class H, class F>
concept has_finally_execute_after_member_function = requires(E ex, H before, F f)
{
  ex.finally_execute_after(before, f);
};

template<class E, class H, class F>
concept has_finally_execute_after_free_function = requires(E ex, H before, F f)
{
  finally_execute_after(ex, before, f);
};


// this is the type of finally_execute_after
struct dispatch_finally_execute_after
{
  // this dispatch path calls the member function
  template<class E, class H, class F>
    requires has_finally_execute_after_member_function<E&&,H&&,F&&>
  constexpr auto operator()(E&& ex, H&& before, F&& f) const
  {
    return std::forward<E>(ex).finally_execute_after(std::forward<H>(before), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class E, class H, class F>
    requires (!has_finally_execute_after_member_function<E&&,H&&,F&&> and has_finally_execute_after_free_function<E&&,H&&,F&&>)
  constexpr auto operator()(E&& ex, H&& before, F&& f) const
  {
    return finally_execute_after(std::forward<E>(ex), std::forward<H>(before), std::forward<F>(f));
  }

  // this dispatch path adapts execute_after
  template<executor E, happening H, std::invocable F>
    requires (!has_finally_execute_after_member_function<E&&,H&&,F&&> and !has_finally_execute_after_free_function<E&&,H&&,F&&>)
  void operator()(E&& ex, H&& before, F&& f) const
  {
    // the happening returned by execute_after is discarded
    execute_after(std::forward<E>(ex), std::forward<H>(before), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_finally_execute_after finally_execute_after;

} // end anonymous namespace


template<class E, class H, class F>
using finally_execute_after_result_t = decltype(ubu::finally_execute_after(std::declval<E>(), std::declval<H>(), std::declval<F>()));


} // end ubu


#include "../../detail/epilogue.hpp"

