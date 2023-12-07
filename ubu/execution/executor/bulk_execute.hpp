#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "../../causality/wait.hpp"
#include "concepts/bulk_executable_on.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class E, class S, class W, class F>
concept has_bulk_execute_member_function = requires(E ex, S shape, W workspace_shape, F func)
{
  ex.bulk_execute_member_function(shape, workspace_shape, func);
};

template<class E, class S, class W, class F>
concept has_bulk_execute_free_function = requires(E ex, S shape, W workspace_shape, F func)
{
  bulk_execute_member_function(ex, shape, workspace_shape, func);
};


// this is the type of bulk_execute
struct dispatch_bulk_execute
{
  // this dispatch path calls executor's member function
  template<class E, class S, class W, class F>
    requires has_bulk_execute_member_function<E&&,S&&,W&&,F&&>
  constexpr void operator()(E&& ex, S&& shape, W&& workspace_shape, F&& func) const
  {
    std::forward<E>(ex).bulk_execute(std::forward<S>(shape), std::forward<W>(workspace_shape), std::forward<F>(func));
  }

  // this dispatch path calls the free function
  template<class E, class S, class W, class F>
    requires (not has_bulk_execute_member_function<E&&,S&&,W&&,F&&>
              and has_bulk_execute_free_function<E&&,S&&,W&&,F&&>)
  constexpr void operator()(E&& ex, S&& shape, W&& workspace_shape, F&& func) const
  {
    bulk_execute(std::forward<E>(ex), std::forward<S>(shape), std::forward<W>(workspace_shape), std::forward<F>(func));
  }

  // this dispatch path calls bulk_execute_after and waits
  template<executor E, coordinate S, weakly_congruent<S> W, bulk_executable_on<E,executor_happening_t<E>,S,W> F>
    requires (not has_bulk_execute_member_function<E,S,W,F>
              and not has_bulk_execute_member_function<E,S,W,F>)
  constexpr void operator()(E ex, S shape, W workspace_shape, F func) const
  {
    wait(bulk_execute_after(ex, initial_happening(ex), shape, workspace_shape, func));
  }
};

} // end detail

inline constexpr detail::dispatch_bulk_execute bulk_execute;

} // end ubu

#include "../../detail/epilogue.hpp"

