#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include <utility>

namespace ubu::detail
{

// bulk_execute_with_workspace_after's dispatch procedure is special
// (cf. allocate_and_zero_after)
//
// bulk execution with a workspace requires allocation of state prior to 
// execution of the user function. In general, an executor may need an
// allocator's help to arrange the allocation of this state.
//
// bulk_execute_with_workspace_after's dispatch allows an executor's customization to
// receive a helper allocator as a parameter. On the other hand, some executors (like CUDA)
// have special optimizations that can create this state without an allocator helper.
// In such cases, these customizations do not need to receive a superfluous allocator parameter.
// Because these superfluous allocator parameters just get in the way, we allow executors
// to customize this operation without including the allocator in their parameter list.
//
// So, dispatch looks for a customization of bulk_execute_with_workspace_after,
// which are, in decreasing priority:
//
// customizations that use the allocator parameter:
// 0. exec.bulk_execute_with_workspace_after(alloc, before, kernel_shape, workspace_shape, f)
// 1. bulk_execute_with_workspace_after(exec, alloc, before, kernel_shape, workspace_shape, f)
// 
// customizations that ignore the allocator parameter:
// 2. exec.bulk_execute_with_workspace_after(before, kernel_shape, workspace_shape, f)
// 3. bulk_execute_with_workspace_after(exec, kernel_shape, workspace_shape, f)

template<class E, class A, class B, class S, class W, class F>
concept has_bulk_execute_with_workspace_after_customization_0 = requires(E executor, A allocator, B before, S kernel_shape, W workspace_shape, F function)
{
  { std::forward<E>(executor).bulk_execute_with_workspace_after(std::forward<A>(allocator), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class A, class B, class S, class W, class F>
concept has_bulk_execute_with_workspace_after_customization_1 = requires(E executor, A allocator, B before, S kernel_shape, W workspace_shape, F function)
{
  { bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(allocator), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class A, class B, class S, class W, class F>
concept has_bulk_execute_with_workspace_after_customization_2 = requires(E executor, B before, S kernel_shape, W workspace_shape, F function)
{
  { std::forward<E>(executor).bulk_execute_with_workspace_after(std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class A, class B, class S, class W, class F>
concept has_bulk_execute_with_workspace_after_customization_3 = requires(E executor, B before, S kernel_shape, W workspace_shape, F function)
{
  { bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};


template<class E, class A, class B, class S, class W, class F>
concept has_custom_bulk_execute_with_workspace_after =
  has_bulk_execute_with_workspace_after_customization_0<E,A,B,S,W,F> or
  has_bulk_execute_with_workspace_after_customization_1<E,A,B,S,W,F> or
  has_bulk_execute_with_workspace_after_customization_2<E,A,B,S,W,F> or
  has_bulk_execute_with_workspace_after_customization_3<E,A,B,S,W,F>
;


template<class E, class A, class B, class S, class W, class F>
  requires has_custom_bulk_execute_with_workspace_after<E&&,A&&,B&&,S&&,W&&,F&&>
constexpr ubu::happening auto custom_bulk_execute_with_workspace_after(E&& ex, A&& alloc, B&& before, S&& kernel_shape, W&& workspace_shape, F&& function)
{
  if constexpr (has_bulk_execute_with_workspace_after_customization_0<E&&,A&&,B&&,S&&,W&&,F&&>)
  {
    // member function
    return std::forward<E>(ex).bulk_execute_with_workspace_after(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
  }
  else if constexpr (has_bulk_execute_with_workspace_after_customization_1<E&&,A&&,B&&,S&&,W&&,F&&>)
  {
    // free function
    return bulk_execute_with_workspace_after(std::forward<E>(ex), std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
  }
  else if constexpr (has_bulk_execute_with_workspace_after_customization_2<E&&,A&&,B&&,S&&,W&&,F&&>)
  {
    // member function, ignored allocator
    return std::forward<E>(ex).bulk_execute_with_workspace_after(std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
  }
  else
  {
    // free function, ignored allocator
    return bulk_execute_with_workspace_after(std::forward<E>(ex), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
  }
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

