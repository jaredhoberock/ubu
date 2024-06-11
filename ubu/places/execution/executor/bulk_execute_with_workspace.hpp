#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/initial_happening.hpp"
#include "../../causality/wait.hpp"
#include "../../memory/allocator/concepts/asynchronous_allocator.hpp"
#include "concepts/bulk_executable_with_workspace_on.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

// XXX bulk_execute_with_workspace needs the more elaborate dispatch procedure similar to bulk_execute_with_workspace_after

template<class E, class A, class S, class W, class F>
concept has_bulk_execute_with_workspace_member_function = requires(E ex, A alloc, S shape, W workspace_shape, F func)
{
  ex.bulk_execute_with_workspace(alloc, shape, workspace_shape, func);
};

template<class E, class A, class S, class W, class F>
concept has_bulk_execute_with_workspace_free_function = requires(E ex, A alloc, S shape, W workspace_shape, F func)
{
  bulk_execute_with_workspace(ex, alloc, shape, workspace_shape, func);
};


// this is the type of bulk_execute_with_workspace
struct dispatch_bulk_execute_with_workspace
{
  // this dispatch path calls executor's member function
  template<class E, class A, class S, class W, class F>
    requires has_bulk_execute_with_workspace_member_function<E&&,A&&,S&&,W&&,F&&>
  constexpr void operator()(E&& ex, A&& alloc, S&& shape, W&& workspace_shape, F&& func) const
  {
    std::forward<E>(ex).bulk_execute_with_workspace(std::forward<A>(alloc), std::forward<S>(shape), std::forward<W>(workspace_shape), std::forward<F>(func));
  }

  // this dispatch path calls the free function
  template<class E, class A, class S, class W, class F>
    requires (not has_bulk_execute_with_workspace_member_function<E&&,A&&,S&&,W&&,F&&>
              and has_bulk_execute_with_workspace_free_function<E&&,A&&,S&&,W&&,F&&>)
  constexpr void operator()(E&& ex, A&& alloc, S&& shape, W&& workspace_shape, F&& func) const
  {
    bulk_execute_with_workspace(std::forward<E>(ex), std::forward<A>(alloc), std::forward<S>(shape), std::forward<W>(workspace_shape), std::forward<F>(func));
  }

  // this dispatch path calls bulk_execute_with_workspace_after and waits
  template<executor E, asynchronous_allocator A, coordinate S, weakly_congruent<S> W, bulk_executable_with_workspace_on<E,A,executor_happening_t<E>,S,W> F>
    requires (not has_bulk_execute_with_workspace_member_function<E,A,S,W,F>
              and not has_bulk_execute_with_workspace_member_function<E,A,S,W,F>)
  constexpr void operator()(E ex, A alloc, S shape, W workspace_shape, F func) const
  {
    wait(bulk_execute_with_workspace_after(ex, alloc, initial_happening(ex), shape, workspace_shape, func));
  }
};

} // end detail

inline constexpr detail::dispatch_bulk_execute_with_workspace bulk_execute_with_workspace;

} // end ubu

#include "../../../detail/epilogue.hpp"

