#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../causality/initial_happening.hpp"
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


template<class E, class F>
concept has_first_execute_member_function = requires(E e, F f)
{
  {e.first_execute(f)} -> happening;
};

template<class E, class F>
concept has_first_execute_free_function = requires(E e, F f)
{
  {first_execute(e, f)} -> happening;
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

  // this dispatch path adapts execute_after
  template<executor E, std::invocable F>
    requires (!has_first_execute_member_function<E&&,F&&> and !has_first_execute_free_function<E&&,F&&>)
  constexpr auto operator()(E&& e, F&& f) const
  {
    return execute_after(std::forward<E>(e), initial_happening(std::forward<E>(e)), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_first_execute first_execute;

} // end anonymous namespace


template<class E, class F>
using first_execute_result_t = decltype(ubu::first_execute(std::declval<E>(), std::declval<F>()));


} // end ubu

#include "../../detail/epilogue.hpp"

