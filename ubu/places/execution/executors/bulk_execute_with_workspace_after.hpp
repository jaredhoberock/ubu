#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../memory/allocators/concepts/asynchronous_allocator.hpp"
#include "concepts/executor.hpp"
#include "detail/custom_bulk_execute_with_workspace_after.hpp"
#include "detail/default_bulk_execute_with_workspace_after.hpp"
#include "detail/one_extending_default_bulk_execute_with_workspace_after.hpp"
#include <cstddef>
#include <utility>


namespace ubu
{
namespace detail
{


// this is the type of bulk_execute_with_workspace_after
struct dispatch_bulk_execute_with_workspace_after
{
  template<class E, class A, class B, class S, class W, class F>
    requires has_custom_bulk_execute_with_workspace_after<E&&,A&&,B&&,S&&,W&&,F&&>
  constexpr happening auto operator()(E&& executor, A&& alloc, B&& before, S&& kernel_shape, W&& workspace_shape, F&& function) const
  {
    return detail::custom_bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(kernel_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
  }

  // dispatch path 4 calls one_extending_bulk_execute_with_workspace_after
  template<executor E, asynchronous_allocator A, happening B, coordinate S, coordinate W, std::invocable<S,executor_workspace_t<E>> F>
    requires (not has_custom_bulk_execute_with_workspace_after<E&&,A&&,B&&,const S&,const W&,F&&>
              and has_one_extending_default_bulk_execute_with_workspace_after<E&&,A&&,B&&,const S&,const W&,F&&>)
  constexpr happening auto operator()(E&& executor, A&& alloc, B&& before, const S& kernel_shape, const W& workspace_shape, F&& function) const
  {
    return one_extending_default_bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(alloc), std::forward<B>(before), kernel_shape, workspace_shape, std::forward<F>(function));
  }
  
  // dispatch path 5 calls default_bulk_execute_with_workspace_after
  template<executor E, asynchronous_allocator A, happening B, coordinate S, std::invocable<S,executor_workspace_t<E>> F>
    requires (    not has_custom_bulk_execute_with_workspace_after<E&&,A&&,B&&,const S&,std::size_t,F&&>
              and not has_one_extending_default_bulk_execute_with_workspace_after<E&&,A&&,B&&,const S&,std::size_t,F&&>
              and has_default_bulk_execute_with_workspace_after<E&&,A&&,B&&,const S&,std::size_t,F&&>)
  constexpr happening auto operator()(E&& executor, A&& allocator, B&& before, const S& kernel_shape, std::size_t workspace_shape, F&& function) const
  {
    return default_bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(allocator), std::forward<B>(before), kernel_shape, workspace_shape, std::forward<F>(function));
  }
};


} // end detail


inline constexpr detail::dispatch_bulk_execute_with_workspace_after bulk_execute_with_workspace_after;

template<class E, class A, class B, class S, class W, class F>
using bulk_execute_with_workspace_after_result_t = decltype(bulk_execute_with_workspace_after(std::declval<E>(), std::declval<A>(), std::declval<B>(), std::declval<S>(), std::declval<W>(), std::declval<F>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

