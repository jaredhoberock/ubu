#pragma once

#include "../../detail/prologue.hpp"

#include "executor.hpp"
#include <concepts>
#include <utility>


namespace ubu
{

namespace detail
{


template<class E, class F, class... Args>
concept has_bind_executable_member_function = requires(E ex, F f, Args... args)
{
  // XXX we should check the result somehow
  ex.bind_executable_member(f, args...);
};

template<class E, class F, class... Args>
concept has_bind_executable_free_function = requires(E ex, F f, Args... args)
{
  // XXX we should check the result somehow
  bind_executable_member(ex, f, args...);
};


// this is the type of bind_executable
struct dispatch_bind_executable
{
  // this dispatch path calls the member function
  template<class E, class F, class... Args>
    requires has_bind_executable_member_function<E&&,F&&,Args&&...>
  constexpr auto operator()(E&& ex, F&& f, Args&&... args) const
  {
    return std::forward<E>(ex).bind_executable(std::forward<F>(f), std::forward<Args>(args)...);
  }

  // this dispatch path calls the free function
  template<class E, class F, class... Args>
    requires has_bind_executable_free_function<E&&,F&&,Args&&...>
  constexpr auto operator()(E&& ex, F&& f, Args&&... args) const
  {
    return bind_executable(std::forward<E>(ex), std::forward<F>(f), std::forward<Args>(args)...);
  }

  template<executor E, class F, class... Args>
    requires (!has_bind_executable_member_function<E&&,F&&,Args&&...>
             and has_bind_executable_free_function<E&&,F&&,Args&&...>)
  constexpr auto operator()(E&& ex, F&& f, Args&&... args) const
  {
    // instead of std::bind, return a lambda here because std::bind's result is
    // not trivially copyable (because std::tuple is not trivially copyable)
    return [f = std::forward<F>(f), ...args = std::forward<Args>(args)](auto&&... additional_args)
    {
      return f(args..., std::forward<decltype(additional_args)>(additional_args)...);
    };
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_bind_executable bind_executable; 

} // end anonymous member


template<class E, class F, class... Args>
using bind_executable_result_t = decltype(ubu::bind_executable(std::declval<E>(), std::declval<F>(), std::declval<Args>()...));


} // end ubu


#include "../../detail/epilogue.hpp"

