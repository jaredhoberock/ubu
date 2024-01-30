#pragma once

#include "../../detail/prologue.hpp"

#include <utility>

namespace ubu
{
namespace detail
{

template<class T, class S>
concept has_coop_dealloca_member_function = requires(T arg, S s)
{
  arg.coop_dealloca(s);
};

template<class T, class S>
concept has_coop_dealloca_free_function = requires(T arg, S s)
{
  coop_dealloca(arg, s);
};

// this is the type of coop_dealloca
struct dispatch_coop_dealloca
{
  // this dispatch path tries the member function
  template<class T, class S>
    requires has_coop_dealloca_member_function<T&&,S&&>
  constexpr void operator()(T&& arg, S&& s) const
  {
    std::forward<T>(arg).coop_dealloca(std::forward<S>(s));
  }

  // this dispatch path tries the free function
  template<class T, class S>
    requires (not has_coop_dealloca_member_function<T&&,S&&>
              and has_coop_dealloca_free_function<T&&,S&&>)
  constexpr void operator()(T&& arg, S&& s) const
  {
    coop_dealloca(std::forward<T>(arg), std::forward<S>(s));
  }
};

} // end detail

constexpr detail::dispatch_coop_dealloca coop_dealloca;

} // end ubu

#include "../../detail/epilogue.hpp"

