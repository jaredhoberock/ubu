#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "concepts/executor.hpp"
#include "detail/default_new_bulk_execute_after.hpp"
#include "traits/executor_happening.hpp"
#include <concepts>
#include <cstddef>
#include <span>
#include <utility>


namespace ubu
{
namespace detail
{


template<class E, class B, class S, class W, class F>
concept has_new_bulk_execute_after_member_function = requires(E executor, B before, S grid_shape, W workspace_shape, F function)
{
  { std::forward<E>(executor).new_bulk_execute_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class B, class S, class W, class F>
concept has_new_bulk_execute_after_free_function = requires(E executor, B before, S grid_shape, W workspace_shape, F function)
{
  { new_bulk_execute_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function)) } -> happening;
};


// this is the type of new_bulk_execute_after
class dispatch_new_bulk_execute_after
{
  public:
    // this dispatch path calls executor's member function
    template<class E, class B, class S, class W, class F>
      requires has_new_bulk_execute_after_member_function<E&&,B&&,S&&,W&&,F&&>
    constexpr happening auto operator()(E&& executor, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return std::forward<E>(executor).new_bulk_execute_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // this dispatch path calls the free function
    template<class E, class B, class S, class W, class F>
      requires (not has_new_bulk_execute_after_member_function<E&&,B&&,S&&,W&&,F&&>
                and has_new_bulk_execute_after_free_function<E&&,B&&,S&&,W&&,F&&>)
    constexpr happening auto operator()(E&& executor, B&& before, S&& grid_shape, W&& workspace_shape, F&& function) const
    {
      return new_bulk_execute_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<W>(workspace_shape), std::forward<F>(function));
    }

    // this dispatch path calls default_new_bulk_execute_after
    template<executor E, happening B, coordinate S, std::regular_invocable<S, std::span<std::byte>> F>
      requires (not has_new_bulk_execute_after_member_function<E&&,B&&,const S&,std::size_t,F&&>
                and not has_new_bulk_execute_after_free_function<E&&,B&&,const S&,std::size_t,F&&>
                and has_default_new_bulk_execute_after<E&&,B&&,const S&,std::size_t,F&&>)
    constexpr executor_happening_t<E> operator()(E&& executor, B&& before, const S& grid_shape, std::size_t workspace_shape, F&& function) const
    {
      return default_new_bulk_execute_after(std::forward<E>(executor), std::forward<B>(before), grid_shape, workspace_shape, std::forward<F>(function));
    }
};


} // end detail


inline constexpr detail::dispatch_new_bulk_execute_after new_bulk_execute_after;

template<class E, class B, class S, class W, class F>
using new_bulk_execute_after_result_t = decltype(new_bulk_execute_after(std::declval<E>(), std::declval<B>(), std::declval<S>(), std::declval<W>(), std::declval<F>()));


} // end ubu

#include "../../detail/epilogue.hpp"

