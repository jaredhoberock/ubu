#pragma once

#include "../../../detail/prologue.hpp"

#include "concepts/layout_like.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_layout_member_function = requires(T arg)
{
  { arg.layout() } -> layout_like;
};

template<class T>
concept has_layout_free_function = requires(T arg)
{
  { layout(arg) } -> layout_like;
};


struct dispatch_layout
{
  template<class T>
    requires has_layout_member_function<T&&>
  constexpr layout_like auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).layout();
  }

  template<class T>
    requires (not has_layout_member_function<T&&>
              and has_layout_free_function<T&&>)
  constexpr layout_like auto operator()(T&& arg) const
  {
    return layout(std::forward<T>(arg));
  }
};

} // end detail


// XXX the decompose CPO makes this CPO superfluous
inline constexpr detail::dispatch_layout layout;


template<class T>
using layout_t = decltype(layout(std::declval<T>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

