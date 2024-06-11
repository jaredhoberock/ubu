#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../../causality/happening.hpp"
#include "../../../memory/allocator/concepts/asynchronous_allocator.hpp"
#include "../../../memory/allocator/allocate_and_zero_after.hpp"
#include "../../../memory/allocator/deallocate_after.hpp"
#include "../../../memory/allocator/traits/allocator_happening_t.hpp"
#include "../concepts/bulk_executable_on.hpp"
#include "../concepts/executor.hpp"
#include "../execute_after.hpp"
#include "../traits/executor_happening.hpp"
#include "../traits/executor_workspace.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>

namespace ubu::detail
{


template<coordinate S, std::invocable<S,std::span<std::byte>> F>
constexpr std::invocable<S> auto make_default_bulk_execute_with_workspace_after_invocable(std::span<std::byte> workspace, F&& function)
{
  return [=, function = std::forward<F>(function)](S coord)
  {
    std::invoke(function, coord, workspace);
  };
}

template<coordinate S, std::invocable<S,std::span<std::byte>> F>
using default_bulk_execute_with_workspace_after_invocable_t = decltype(make_default_bulk_execute_with_workspace_after_invocable<S>(std::declval<std::span<std::byte>>(), std::declval<F>()));


// XXX for now, this default version of bulk_execute_with_workspace_after will only work for executors whose workspace type is std::span
template<class E>
concept executor_with_simple_workspace = executor<E> and std::same_as<std::span<std::byte>, executor_workspace_t<E>>;

template<executor_with_simple_workspace E, asynchronous_allocator A, happening B, coordinate S, std::invocable<S,executor_workspace_t<E>> F>
  requires bulk_executable_on<default_bulk_execute_with_workspace_after_invocable_t<S,F>, E, allocator_happening_t<A>, S>
allocator_happening_t<A> default_bulk_execute_with_workspace_after(const E& ex, const A& alloc, B&& before, const S& shape, std::size_t workspace_size, F&& function)
{
  // XXX this function should call detail::construct_workspaces_after(ex, alloc, before, shape, workspace_size)
  //     which would check executor_workspace_shape_t and construct buffers & barriers as necessary
  //     it could return a tensor of workspaces such that each coordinate's workspace would be found like so:
  //
  //         workspace auto ws = tensor_of_workspaces[coord];
  //
  //     then, we'd follow it up with delete_workspaces_after

  // allocate a workspace
  // XXX concurrent executors do not need their workspace zeroed
  auto [workspace_ready, workspace_ptr] = allocate_and_zero_after<std::byte>(alloc, ex, std::forward<B>(before), workspace_size);

  // execute the user function after the workspace is allocated
  std::span<std::byte> workspace(workspace_ptr, workspace_size);
  auto execution_finished = bulk_execute_after(ex, std::move(workspace_ready), shape, make_default_bulk_execute_with_workspace_after_invocable<S>(workspace, std::forward<F>(function)));

  // deallocate the workspace after the user function executes
  return deallocate_after(alloc, std::move(execution_finished), workspace_ptr, workspace_size);
}


template<class E, class A, class B, class S, class W, class F>
concept has_default_bulk_execute_with_workspace_after = requires(E ex, A alloc, B before, S shape, W workspace_shape, F f)
{
  { default_bulk_execute_with_workspace_after(std::forward<E>(ex), std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape), std::forward<W>(workspace_shape), std::forward<F>(f)) } -> happening;
};


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

