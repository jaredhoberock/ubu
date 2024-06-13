#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/pointers/pointer_like.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T, class S>
concept has_coop_alloca_member_function = requires(T arg, S s)
{
  { arg.coop_alloca(s) } -> pointer_like;
};

template<class T, class S>
concept has_coop_alloca_free_function = requires(T arg, S s)
{
  { coop_alloca(arg, s) } -> pointer_like;
};

// this is the type of coop_alloca
struct dispatch_coop_alloca
{
  // this dispatch path tries the member function
  template<class T, class S>
    requires has_coop_alloca_member_function<T&&,S&&>
  constexpr pointer_like auto operator()(T&& arg, S&& s) const
  {
    return std::forward<T>(arg).coop_alloca(std::forward<S>(s));
  }

  // this dispatch path tries the free function
  template<class T, class S>
    requires (not has_coop_alloca_member_function<T&&,S&&>
              and has_coop_alloca_free_function<T&&,S&&>)
  constexpr pointer_like auto operator()(T&& arg, S&& s) const
  {
    return coop_alloca(std::forward<T>(arg), std::forward<S>(s));
  }
};

} // end detail

constexpr detail::dispatch_coop_alloca coop_alloca;

} // end ubu

#include "../../detail/epilogue.hpp"

