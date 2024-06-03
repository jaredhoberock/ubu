#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include "../../../tensor/coordinate/concepts/strictly_subdimensional.hpp"
#include "../../../tensor/coordinate/traits/default_coordinate.hpp"
#include "../../../tensor/views/layout/truncating_layout.hpp"
#include "../concepts/executor.hpp"
#include "../traits/executor_happening.hpp"
#include "../traits/executor_shape.hpp"
#include <concepts>
#include <functional>
#include <utility>

namespace ubu::detail
{


template<class CPO, class E, class H, class S, class F>
concept has_one_extending_default_bulk_execute_after =
  executor<E>
  and happening<H>
  and strictly_subdimensional<S,executor_shape_t<E>>
  and std::invocable<F,default_coordinate_t<S>>
  and requires(CPO bulk_execute_after_cpo, E ex, H before, const S& user_shape)
  {
    { bulk_execute_after_cpo(std::forward<E>(ex), std::forward<H>(before), truncating_layout<executor_shape_t<E>,S>(user_shape).shape(), [](auto){}) } -> happening;
  }
;


// this default implementation of bulk_execute_after one-extends the user's shape to make it
// congruent with executor_shape_t<E> and then calls the lower-level function via the bulk_execute_after CPO
template<class CPO, executor E, happening H, strictly_subdimensional<executor_shape_t<E>> S, std::invocable<default_coordinate_t<S>> F>
  requires has_one_extending_default_bulk_execute_after<CPO,E&&,H&&,S,F>
executor_happening_t<E> one_extending_default_bulk_execute_after(CPO bulk_execute_after_cpo, E&& ex, H&& before, const S& user_shape, F user_function)
{
  // we'll one-extend the user's requested shape and
  // map native coordinates produced by the executor
  // to the user's coordinate type via truncation
  truncating_layout<executor_shape_t<E>,S> layout(user_shape);

  return bulk_execute_after_cpo(std::forward<E>(ex), std::forward<H>(before), layout.shape(), [=](auto native_coord)
  {
    std::invoke(user_function, layout[native_coord]);
  });
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

