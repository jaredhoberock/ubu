#pragma once

#include "../../detail/prologue.hpp"

#include "../../memory/pointer/pointer_like.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T, class S>
concept has_push_shared_buffer_member_function = requires(T arg, S s)
{
  { arg.push_shared_buffer(s) } -> pointer_like;
};

template<class T, class S>
concept has_push_shared_buffer_free_function = requires(T arg, S s)
{
  { push_shared_buffer(arg, s) } -> pointer_like;
};

// this is the type of push_shared_buffer
struct dispatch_push_shared_buffer
{
  // this dispatch path tries the member function
  template<class T, class S>
    requires has_push_shared_buffer_member_function<T&&,S&&>
  constexpr pointer_like auto operator()(T&& arg, S&& s) const
  {
    return std::forward<T>(arg).push_shared_buffer(std::forward<S>(s));
  }

  // this dispatch path tries the free function
  template<class T, class S>
    requires (not has_push_shared_buffer_member_function<T&&,S&&>
              and has_push_shared_buffer_free_function<T&&,S&&>)
  constexpr pointer_like auto operator()(T&& arg, S&& s) const
  {
    return push_shared_buffer(std::forward<T>(arg), std::forward<S>(s));
  }
};

} // end detail

constexpr detail::dispatch_push_shared_buffer push_shared_buffer;

} // end ubu

#include "../../detail/epilogue.hpp"

