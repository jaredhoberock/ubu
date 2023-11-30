#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/first_cause.hpp"
#include "../../causality/wait.hpp"
#include "../../grid/coordinate/compare.hpp"
#include "../../grid/coordinate/coordinate.hpp"
#include "../../grid/layout/layout.hpp"
#include "../../grid/shape.hpp"
#include "bulk_execute_after.hpp"
#include "concepts/executor.hpp"
#include "kernel_layout.hpp"
#include "traits/executor_coordinate.hpp"
#include <concepts>
#include <utility>


namespace ubu
{

namespace detail
{

template<class E, class S, class F>
concept can_bulk_execute_after = requires(E ex, S shape, F f)
{
  wait(bulk_execute_after(ex, first_cause(ex), shape, f));
};


// when bulk_execute_after can be called directly with the arguments
template<executor E, coordinate S, std::invocable<S> F>
  requires can_bulk_execute_after<E,S&,F&&>
void default_execute_kernel(E ex, const S& shape, F&& f)
{
  // XXX it would be more convenient to call a bulk_execute CPO
  auto finished = bulk_execute_after(ex, first_cause(ex), shape, std::forward<F>(f));

  wait(finished);
}


// when bulk_execute_after cannot be called directly with the arguments
template<executor E, coordinate S, std::invocable<S> F>
  requires (not can_bulk_execute_after<E,S&,F&&>)
void default_execute_kernel(E ex, const S& user_shape, F&& f)
{
  // to_user_coord is a layout which maps a coordinate originating from
  // an executor to a coordinate within the user's requested shape
  layout auto to_user_coord = kernel_layout(ex, user_shape, std::forward<F>(f));

  // XXX it would be more convenient to call a bulk_execute CPO
  auto finished = bulk_execute_after(ex, first_cause(ex), shape(to_user_coord), [=](const executor_coordinate_t<E>& ex_coord)
  {
    S user_coord = to_user_coord[ex_coord];
    if(is_below(user_coord, user_shape))
    {
      f(user_coord);
    }
  });

  wait(finished);
}


template<class E, class S, class F>
concept has_execute_kernel_member_function = requires(E ex, S shape, F f)
{
  {ex.execute_kernel(shape, f)} -> std::same_as<void>;
};

template<class E, class S, class F>
concept has_execute_kernel_free_function = requires(E ex, S shape, F f)
{
  {execute_kernel(ex, shape, f)} -> std::same_as<void>;
};


// this is the type of execute_kernel
struct dispatch_execute_kernel
{
  // this dispatch path calls the member function
  template<class E, class S, class F>
    requires has_execute_kernel_member_function<E&&,S&&,F&&>
  constexpr void operator()(E&& ex, S&& shape, F&& f) const
  {
    std::forward<E>(ex).execute_kernel(std::forward<S>(shape), std::forward<F>(f));
  }

  // this dispatch path calls the free function
  template<class E, class S, class F>
    requires (not has_execute_kernel_member_function<E&&,S&&,F&&>
              and has_execute_kernel_free_function<E&&,S&&,F&&>)
  constexpr void operator()(E&& ex, S&& shape, F&& f) const
  {
    execute_kernel(std::forward<E>(ex), std::forward<S>(shape), std::forward<F>(f));
  }

  // this dispatch path calls the default implementation
  template<executor E, coordinate S, std::invocable<S> F>
    requires (not has_execute_kernel_member_function<E&&,S&&,F&&>
              and not has_execute_kernel_free_function<E&&,S&&,F&&>)
  constexpr void operator()(E&& ex, S&& shape, F&& f) const
  {
    default_execute_kernel(std::forward<E>(ex), std::forward<S>(shape), std::forward<F>(f));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_execute_kernel execute_kernel;

} // end anonymous namespace


} // end ubu


#include "../../detail/epilogue.hpp"

