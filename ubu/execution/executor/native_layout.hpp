#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/layout/column_major.hpp"
#include "../../tensor/layout/layout.hpp"
#include "concepts/executor.hpp"
#include "traits/executor_coordinate.hpp"
#include <utility>

// The purpose of this CPO is for objects which produce coordinates to be able to describe a linear ordering with a "native" layout 
//
// For example, executors produce coordinates which are passed to invocables
// We may also need to describe the layout of objects in memory produced by allocators
//
// XXX this header ought to be organized underneath tensor/layout/
//     the problem is that we need to mention executor_coordinate_t in the default dispatch path below
//     we either need to find a way to eliminate this CPO, or we need to be able to find a generic implementation of the default path

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

  // this dispatch path assumes a column-major layout for executors
  template<executor E>
    requires (not has_native_layout_member_function<const E&, const executor_coordinate_t<E>&>
              and not has_native_layout_free_function<const E&, const executor_coordinate_t<E>&>)
  constexpr layout auto operator()(const E&, const executor_coordinate_t<E>& shape) const
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

