#pragma once

#include "../../detail/prologue.hpp"
#include "barrier_like.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_barrier_member_variable = requires(T arg)
{
// XXX WAR circle bug
#if defined(__circle_lang__)
  arg.barrier;
  requires requires(decltype(arg.barrier) bar)
  {
    { bar } -> barrier_like;
  };
#else
  { arg.barrier } -> barrier_like;
#endif
};

template<class T>
concept has_get_barrier_member_function = requires(T arg)
{
  { arg.get_barrier() } -> barrier_like;
};

template<class T>
concept has_get_barrier_free_function = requires(T arg)
{
  { get_barrier(arg) } -> barrier_like;
};


struct dispatch_get_barrier
{
  template<class T>
    requires has_barrier_member_variable<T&&>
  constexpr barrier_like decltype(auto) operator()(T&& arg) const
  {
    // return a reference to the member
    return (std::forward<T>(arg).barrier);
  }

  template<class T>
    requires (not has_barrier_member_variable<T&&>
              and has_get_barrier_member_function<T&&>)
  constexpr barrier_like decltype(auto) operator()(T&& arg) const
  {
    return std::forward<T>(arg).get_barrier();
  }

  template<class T>
    requires (not has_barrier_member_variable<T&&>
              and not has_get_barrier_member_function<T&&>
              and has_get_barrier_free_function<T&&>)
  constexpr barrier_like decltype(auto) operator()(T&& arg) const
  {
    return get_barrier(std::forward<T>(arg));
  }
};

} // end detail


inline constexpr detail::dispatch_get_barrier get_barrier;

template<class T>
using barrier_t = std::remove_cvref_t<decltype(get_barrier(std::declval<T>()))>;


} // end ubu

#include "../../detail/epilogue.hpp"

