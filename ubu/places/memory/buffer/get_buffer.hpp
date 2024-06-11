#pragma once

#include "../../../detail/prologue.hpp"
#include "buffer_like.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_buffer_member_variable = requires(T arg)
{
  arg.buffer;
  buffer_like<decltype(arg.buffer)>;
};

template<class T>
concept has_get_buffer_member_function = requires(T arg)
{
  { arg.get_buffer() } -> buffer_like;
};

template<class T>
concept has_get_buffer_free_function = requires(T arg)
{
  { get_buffer(arg) } -> buffer_like;
};


struct dispatch_get_buffer
{
  template<buffer_like B>
  constexpr B operator()(B buf) const
  {
    return buf;
  }

  template<class T>
    requires has_buffer_member_variable<T&&>
  constexpr buffer_like auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).buffer;
  }

  template<class T>
    requires (not has_buffer_member_variable<T&&>
              and has_get_buffer_member_function<T&&>)
  constexpr buffer_like auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).get_buffer();
  }

  template<class T>
    requires (not has_buffer_member_variable<T&&>
              and not has_get_buffer_member_function<T&&>
              and has_get_buffer_free_function<T&&>)
  constexpr buffer_like auto operator()(T&& arg) const
  {
    return get_buffer(std::forward<T>(arg));
  }
};

} // end detail


inline constexpr detail::dispatch_get_buffer get_buffer;

template<class T>
using buffer_t = std::remove_cvref_t<decltype(get_buffer(std::declval<T>()))>;

} // end ubu

#include "../../../detail/epilogue.hpp"

