#pragma once

#include "../../../detail/prologue.hpp"

#include "../../shapes/shape.hpp"
#include "column_major.hpp"
#include "layout.hpp"
#include <utility>

// The purpose of this CPO is for objects which produce coordinates to be able to describe a linear ordering with a "native" layout 
//
// For example, executors produce coordinates which are passed to invocables
// We may also need to describe the layout of objects in memory produced by allocators

namespace ubu
{
namespace detail
{

template<class T, class S>
concept has_native_layout_member_function = 
  coordinate<S>
  and requires(T arg, S shape)
  {
    { arg.native_layout(shape) } -> layout;
    { shape(arg.native_layout(shape)) } -> congruent<S>;
  }
;

template<class T, class S>
concept has_native_layout_free_function =
  coordinate<S>
  and requires(T arg, S shape)
  {
    { native_layout(arg, shape) } -> layout;
    { shape(arg.native_layout(shape)) } -> congruent<S>;
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

  // this default dispatch path produces a column-major layout
  template<class T, coordinate S>
    requires (not has_native_layout_member_function<T&&, const S&>
              and not has_native_layout_free_function<T&&, const S&>)
  constexpr layout auto operator()(T&&, const S& shape) const
  {
    // produce a column-major ordering by default
    return column_major(shape);
  }
};

} // end detail

namespace
{

// XXX this should maybe be named default_layout
constexpr detail::dispatch_native_layout native_layout;

} // end anonymous namespace

} // end ubu

#include "../../../detail/epilogue.hpp"

