#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../memory/allocators/concepts/asynchronous_allocator.hpp"
#include "../../memory/allocators/traits/allocator_happening_t.hpp"
#include "concepts/executor.hpp"
#include "detail/default_bulk_execute_with_workspace_after.hpp"
#include "detail/one_extending_default_bulk_execute_with_workspace_after.hpp"
#include "traits/executor_happening.hpp"
#include "traits/executor_workspace.hpp"
#include <concepts>
#include <cstddef>
#include <span>
#include <utility>


namespace ubu
{
namespace detail
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
// 0. exec.bulk_execute_with_workspace_after(alloc, before, shape, workspace_shape, f)
// 1. bulk_execute_with_workspace_after(exec, alloc, before, shape, workspace_shape, f)
// 
// customizations that ignore the allocator parameter:
// 2. exec.bulk_execute_with_workspace_after(before, shape, workspace_shape, f)
// 3. bulk_execute_with_workspace_after(exec, shape, workspace_shape, f)
//
// if dispatch fails to find a customization, it uses a default:
// 4. one_extending_bulk_execute_with_workspace_after(args...), or
// 5. allocate_and_zero_after(alloc, ...) then bulk_execute_after(ex, ...) then deallocate_after(alloc, ...)
namespace dispatch_bulk_execute_with_workspace_after
{


template<class E, class A, class B, class S, class W, class F>
concept has_customization_0 = requires(E executor, A allocator, B before, S grid_shape, W workspace_shape, F function)
{
  { std::forward<E>(executor).bulk_execute_with_workspace_after(std::forward<A>(allocator), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class A, class B, class S, class W, class F>
concept has_customization_1 = requires(E executor, A allocator, B before, S grid_shape, W workspace_shape, F function)
{
  { bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(allocator), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class A, class B, class S, class W, class F>
concept has_customization_2 = requires(E executor, B before, S grid_shape, W workspace_shape, F function)
{
  { std::forward<E>(executor).bulk_execute_with_workspace_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class A, class B, class S, class W, class F>
concept has_customization_3 = requires(E executor, B before, S grid_shape, W workspace_shape, F function)
{
  { bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};


// this is the type of bulk_execute_with_workspace_after
class cpo
{
  public:
    // dispatch path 0 calls executor's member function with the allocator
    template<class E, class A, class B, class S, class W, class F>
      requires has_customization_0<E&&,A&&,B&&,S&&,W&&,F&&>
    constexpr happening auto operator()(E&& executor, A&& allocator, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_with_workspace_after(std::forward<A>(allocator), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // dispatch path 1 calls the free function with the allocator
    template<class E, class A, class B, class S, class W, class F>
      requires (not has_customization_0<E&&,A&&,B&&,S&&,W&&,F&&>
                and has_customization_1<E&&,A&&,B&&,S&&,W&&,F&&>)
    constexpr happening auto operator()(E&& executor, A&& allocator, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(allocator), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // dispatch path 2 calls executor's member function without the allocator
    template<class E, class A, class B, class S, class W, class F>
      requires (    not has_customization_0<E&&,A&&,B&&,S&&,W&&,F&&>
                and not has_customization_1<E&&,A&&,B&&,S&&,W&&,F&&>
                    and has_customization_2<E&&,A&&,B&&,S&&,W&&,F&&>)
    constexpr happening auto operator()(E&& executor, A&&, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_with_workspace_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // dispatch path 3 calls the free function without the allocator
    template<class E, class A, class B, class S, class W, class F>
      requires (    not has_customization_0<E&&,A&&,B&&,S&&,W&&,F&&>
                and not has_customization_1<E&&,A&&,B&&,S&&,W&&,F&&>
                and not has_customization_2<E&&,A&&,B&&,S&&,W&&,F&&>
                    and has_customization_3<E&&,A&&,B&&,S&&,W&&,F&&>)
    constexpr happening auto operator()(E&& executor, A&&, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // dispatch path 4 calls one_extending_bulk_execute_with_workspace_after
    template<executor E, asynchronous_allocator A, happening B, coordinate S, coordinate W, std::invocable<S,executor_workspace_t<E>> F>
      requires (    not has_customization_0<E&&,A&&,B&&,const S&,const W&,F&&>
                and not has_customization_1<E&&,A&&,B&&,const S&,const W&,F&&>
                and not has_customization_2<E&&,A&&,B&&,const S&,const W&,F&&>
                and not has_customization_3<E&&,A&&,B&&,const S&,const W&,F&&>
                    and has_one_extending_default_bulk_execute_with_workspace_after<cpo,E&&,A&&,B&&,const S&,const W&,F&&>)
    constexpr happening auto operator()(E&& executor, A&& alloc, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return one_extending_default_bulk_execute_with_workspace_after(*this, std::forward<E>(executor), std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // dispatch path 5 calls default_bulk_execute_with_workspace_after
    template<executor E, asynchronous_allocator A, happening B, coordinate S, std::invocable<S,executor_workspace_t<E>> F>
      requires (    not has_customization_0<E&&,A&&,B&&,const S&,std::size_t,F&&>
                and not has_customization_1<E&&,A&&,B&&,const S&,std::size_t,F&&>
                and not has_customization_2<E&&,A&&,B&&,const S&,std::size_t,F&&>
                and not has_customization_3<E&&,A&&,B&&,const S&,std::size_t,F&&>
                and not has_one_extending_default_bulk_execute_with_workspace_after<cpo,E&&,A&&,B&&,const S&,std::size_t,F&&>
                    and has_default_bulk_execute_with_workspace_after<E&&,A&&,B&&,const S&,std::size_t,F&&>)
    constexpr allocator_happening_t<A> operator()(E&& executor, A&& allocator, B&& before, const S& grid_shape, std::size_t workspace_shape, F&& function) const
    {
      return default_bulk_execute_with_workspace_after(std::forward<E>(executor), std::forward<A>(allocator), std::forward<B>(before), grid_shape, workspace_shape, std::forward<F>(function));
    }
};


} // end dispatch_bulk_execute_with_workspace_after
} // end detail


inline constexpr detail::dispatch_bulk_execute_with_workspace_after::cpo bulk_execute_with_workspace_after;

template<class E, class A, class B, class S, class W, class F>
using bulk_execute_with_workspace_after_result_t = decltype(bulk_execute_with_workspace_after(std::declval<E>(), std::declval<A>(), std::declval<B>(), std::declval<S>(), std::declval<W>(), std::declval<F>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

