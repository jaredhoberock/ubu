#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include <utility>

// the purpose of custom_bulk_execute_after is to simply call either
//
// 1. ex.bulk_execute_after(args...), or
// 2. bulk_execute_after(ex, args...)
//
// if one of the two calls is well-formed.
// in other words, custom_bulk_execute_after calls the executor's customization of bulk_execute_after, if one exists

namespace ubu::detail
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

template<class E, class B, class S, class F>
concept has_custom_bulk_execute_after =
  has_bulk_execute_after_member_function<E,B,S,F>
  or has_bulk_execute_after_free_function<E,B,S,F>
;

template<class E, class B, class S, class F>
  requires has_custom_bulk_execute_after<E&&,B&&,S&&,F&&>
constexpr happening auto custom_bulk_execute_after(E&& ex, B&& before, S&& grid_shape, F&& function)
{
  if constexpr (has_bulk_execute_after_member_function<E&&,B&&,S&&,F&&>)
  {
    return std::forward<E>(ex).bulk_execute_after(std::forward<B>(before), std::forward<S>(grid_shape), std::forward<F>(function));
  }
  else
  {
    return bulk_execute_after(std::forward<E>(ex), std::forward<B>(before), std::forward<S>(grid_shape), std::forward<F>(function));
  }
}

} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

