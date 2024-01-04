#pragma once

#include "../../detail/prologue.hpp"

#include "../../grid/coordinate/concepts/coordinate.hpp"
#include "concepts/executor.hpp"
#include "detail/default_execute_kernel.hpp"
#include <concepts>
#include <utility>


namespace ubu
{

namespace detail
{

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

