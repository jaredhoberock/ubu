#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


template<class E, class H, class F>
concept has_execute_after_member_function = requires(E ex, H before, F f)
{
  {ex.execute_after(before, f)} -> happening;
};

template<class E, class H, class F>
concept has_execute_after_free_function = requires(E ex, H before, F f)
{
  {execute_after(ex, before, f)} -> happening;
};


// this is the type of execute_after
struct dispatch_execute_after
{
  // this dispatch path calls the member function
  template<class E, class H, class F>
    requires has_execute_after_member_function<E&&,H&&,F&&>
  constexpr auto operator()(E&& ex, H&& before, F&& f) const
  {
    return std::forward<E>(ex).execute_after(std::forward<H>(before), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class E, class H, class F>
    requires (!has_execute_after_member_function<E&&,H&&,F&&> and has_execute_after_free_function<E&&,H&&,F&&>)
  constexpr auto operator()(E&& ex, H&& before, F&& f) const
  {
    return execute_after(std::forward<E>(ex), std::forward<H>(before), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_execute_after execute_after;

} // end anonymous namespace


template<class E, class H, class F>
using execute_after_result_t = decltype(ubu::execute_after(std::declval<E>(), std::declval<H>(), std::declval<F>()));


} // end ubu


#include "../../../detail/epilogue.hpp"

