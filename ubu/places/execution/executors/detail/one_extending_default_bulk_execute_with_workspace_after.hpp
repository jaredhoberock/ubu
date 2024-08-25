#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/subdimensional.hpp"
#include "../../../../tensors/coordinates/traits/default_coordinate.hpp"
#include "../../../../tensors/views/layouts/truncating_layout.hpp"
#include "../../../causality/happening.hpp"
#include "../concepts/executor.hpp"
#include "../traits/executor_happening.hpp"
#include "../traits/executor_shape.hpp"
#include "custom_bulk_execute_with_workspace_after.hpp"
#include <concepts>
#include <functional>
#include <utility>

namespace ubu::detail
{


template<class E, class A, class H, class S, class W, class F>
concept has_one_extending_default_bulk_execute_with_workspace_after =
  executor<E>
  and asynchronous_allocator<A>
  and happening<H>
  and subdimensional<S,executor_shape_t<E>>
  and congruent<W,executor_workspace_shape_t<E>>
  and equal_rank<S,W>
  and std::invocable<F,default_coordinate_t<S>,executor_workspace_t<E>>
  and requires(E ex, A alloc, H before, const S& user_shape, const W& workspace_shape)
  {
    { detail::custom_bulk_execute_with_workspace_after(std::forward<E>(ex), std::forward<A>(alloc), std::forward<H>(before), truncating_layout<executor_shape_t<E>,S>(user_shape).shape(), workspace_shape, [](auto,auto){}) } -> happening;
  }
;


// this default implementation of bulk_execute_with_workspace_after one-extends the user's shape to make it
// congruent with executor_shape_t<E> and then calls the executor's customization of bulk_execute_with_workspace_after
template<executor E,
         asynchronous_allocator A,
         happening H,
         subdimensional<executor_shape_t<E>> S,
         congruent<executor_workspace_shape_t<E>> W,
         std::invocable<default_coordinate_t<S>,executor_workspace_t<E>> F
        >
  requires (equal_rank<S,W> and has_one_extending_default_bulk_execute_with_workspace_after<E&&,A&&,H&&,S,W,F>)
executor_happening_t<E> one_extending_default_bulk_execute_with_workspace_after(E&& ex, A&& alloc, H&& before, const S& user_shape, const W& workspace_shape, F user_function)
{
  // we'll one-extend the user's requested shape and
  // map native coordinates produced by the executor
  // to the user's coordinate type via truncation
  truncating_layout<executor_shape_t<E>,S> layout(user_shape);

  return detail::custom_bulk_execute_with_workspace_after(std::forward<E>(ex), std::forward<A>(alloc), std::forward<H>(before), layout.shape(), workspace_shape, [=](auto native_coord, auto workspace)
  {
    std::invoke(user_function, layout[native_coord], workspace);
  });
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

