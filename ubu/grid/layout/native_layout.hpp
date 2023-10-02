#pragma once

#include "../../detail/prologue.hpp"

#include "column_major.hpp"
#include "layout.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

// XXX should also insist that the resulting layout's shape_t is S
template<class T, class S>
concept has_native_layout_member_function = 
  coordinate<S>
  and requires(T arg, S shape)
  {
    { arg.native_layout(shape) } -> layout;
  }
;

// XXX should also insist that the resulting layout's shape_t is S
template<class T, class S>
concept has_native_layout_free_function =
  coordinate<S>
  and requires(T arg, S shape)
  {
    { native_layout(arg, shape) } -> layout;
  }
;

struct dispatch_native_layout
{
  template<class T, class S>
    requires has_native_layout_member_function<T&&,S&&>
  constexpr layout auto operator()(T&& arg, S&& shape) const
  {
    return std::forward<T>(arg).native_layout(std::forward<S>(shape));
  }

  template<class T, class S>
    requires (not has_native_layout_member_function<T&&,S&&>
              and has_native_layout_free_function<T&&,S&&>)
  constexpr layout auto operator()(T&& arg, S&& shape) const
  {
    return native_layout(std::forward<T>(arg), std::forward<S>(shape));
  }

  template<class T>
    requires (not has_compose_member_function<const T&,typename T::coordinate_type&>
              and not has_compose_free_function<const T&,typename T::coordinate_type&>)
  constexpr layout auto operator()(const T&, typename T::coordinate_type shape) const
  {
    // assume a column-major ordering by default
    return column_major(shape);
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_native_layout native_layout;

} // end anonymous namespace

} // end ubu

#include "../../detail/epilogue.hpp"

