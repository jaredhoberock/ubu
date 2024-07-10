#pragma once

#include "../../detail/prologue.hpp"
#include "pointers/pointer_like.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_data_member_function = requires(T arg)
{
  { std::forward<T>(arg).data() } -> pointer_like;
};

template<class T>
concept has_data_free_function = requires(T arg)
{
  { data(std::forward<T>(arg)) } -> pointer_like;
};


struct dispatch_data
{
  template<class T>
    requires has_data_member_function<T&&>
  constexpr pointer_like auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).data();
  }

  template<class T>
    requires (not has_data_member_function<T&&>
              and has_data_free_function<T&&>)
  constexpr pointer_like auto operator()(T&& arg) const
  {
    return data(std::forward<T>(arg));
  }
};


} // end detail

inline constexpr detail::dispatch_data data;

template<class T>
using data_t = decltype(data(std::declval<T>()));

} // end ubu

#include "../../detail/epilogue.hpp"

