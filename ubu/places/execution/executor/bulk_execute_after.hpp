#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "concepts/executor.hpp"
#include "traits/executor_happening.hpp"
#include "detail/one_extending_default_bulk_execute_after.hpp"
#include "detail/sequential_default_bulk_execute_after.hpp"
#include <concepts>
#include <utility>


namespace ubu
{
namespace detail
{


template<class E, class B, class S, class F>
concept has_bulk_execute_after_member_function = requires(E executor, B before, S grid_shape, F function)
{
  { std::forward<E>(executor).bulk_execute_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<F>(function)) } -> happening;
};

template<class E, class B, class S, class F>
concept has_bulk_execute_after_free_function = requires(E executor, B before, S grid_shape, F function)
{
  { bulk_execute_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<F>(function)) } -> happening;
};


// this is the type of bulk_execute_after
class dispatch_bulk_execute_after
{
  public:
    // this dispatch path calls executor's member function
    template<class E, class B, class S, class F>
      requires has_bulk_execute_after_member_function<E&&,B&&,S&&,F&&>
    constexpr happening auto operator()(E&& executor, B&& before, S&& grid_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<F>(function));
    }

    // this dispatch path calls the free function
    template<class E, class B, class S, class F>
      requires (not has_bulk_execute_after_member_function<E&&,B&&,S&&,F&&>
                and has_bulk_execute_after_free_function<E&&,B&&,S&&,F&&>)
    constexpr happening auto operator()(E&& executor, B&& before, S&& grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<E>(executor), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<F>(function));
    }

    // this dispatch path calls one_extending_default_bulk_execute_after
    template<executor E, happening B, coordinate S, std::invocable<S> F>
      requires (not has_bulk_execute_after_member_function<E&&,B&&,const S&,F&&>
                and not has_bulk_execute_after_free_function<E&&,B&&,const S&,F&&>
                and has_one_extending_default_bulk_execute_after<dispatch_bulk_execute_after,E&&,B&&,const S&,F&&>)
    constexpr executor_happening_t<E> operator()(E&& executor, B&& before, const S& grid_shape, F&& function) const
    {
      return one_extending_default_bulk_execute_after(*this, std::forward<E>(executor), std::forward<B>(before), grid_shape, std::forward<F>(function));
    }

    // this dispatch path calls sequential_default_bulk_execute_after
    template<executor E, happening B, coordinate S, std::invocable<S> F>
      requires (not has_bulk_execute_after_member_function<E&&,B&&,const S&,F&&>
                and not has_bulk_execute_after_free_function<E&&,B&&,const S&,F&&>
                and not has_one_extending_default_bulk_execute_after<dispatch_bulk_execute_after,E&&,B&&,const S&,F&&>
                and has_sequential_default_bulk_execute_after<E&&,B&&,const S&,F&&>)
    constexpr executor_happening_t<E> operator()(E&& executor, B&& before, const S& grid_shape, F&& function) const
    {
      return sequential_default_bulk_execute_after(std::forward<E>(executor), std::forward<B>(before), grid_shape, std::forward<F>(function));
    }
};


} // end detail


inline constexpr detail::dispatch_bulk_execute_after bulk_execute_after;

template<class E, class B, class S, class F>
using bulk_execute_after_result_t = decltype(bulk_execute_after(std::declval<E>(), std::declval<B>(), std::declval<S>(), std::declval<F>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

