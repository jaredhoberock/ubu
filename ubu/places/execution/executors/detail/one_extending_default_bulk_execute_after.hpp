#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/strictly_subdimensional.hpp"
#include "../../../../tensors/coordinates/traits/default_coordinate.hpp"
#include "../../../../tensors/views/layouts/truncating_layout.hpp"
#include "../../../causality/happening.hpp"
#include "../concepts/executor.hpp"
#include "../traits/executor_happening.hpp"
#include "../traits/executor_shape.hpp"
#include "custom_bulk_execute_after.hpp"
#include <concepts>
#include <functional>
#include <utility>

namespace ubu::detail
{


template<class E, class H, class S, class F>
concept has_one_extending_default_bulk_execute_after =
  executor<E>
  and happening<H>
  and strictly_subdimensional<S,executor_shape_t<E>> // XXX it think maybe we just want weakly_congruent. strictly_subdimensional seems superfluous
  and std::invocable<F,default_coordinate_t<S>>
  and requires(E ex, H before, const S& user_shape)
  {
    { detail::custom_bulk_execute_after(std::forward<E>(ex), std::forward<H>(before), truncating_layout<executor_shape_t<E>,S>(user_shape).shape(), [](auto){}) } -> happening;
  }
;


// the purpose of this default path for bulk_execute_after is to perform simple conversions on the user's arguments (right now, just user_shape)
// to match the executor's expectations. Then, it simply forwards the arguments along to the executor's customization of bulk_execute_after
//
// The conversion performed on user_shape simply makes it congruent with executor_shape_t<E> by one-extending the value of the user's shape
//
// in principle, we could also convert the before argument into the executor's happening type, if the type of before doesn't match executor_happening_t<E>
template<executor E, happening H, strictly_subdimensional<executor_shape_t<E>> S, std::invocable<default_coordinate_t<S>> F>
  requires has_one_extending_default_bulk_execute_after<E&&,H&&,S,F>
executor_happening_t<E> one_extending_default_bulk_execute_after(E&& ex, H&& before, const S& user_shape, F user_function)
{
  // we'll one-extend the user's requested shape and
  // map native coordinates produced by the executor
  // to the user's coordinate type via truncation
  truncating_layout<executor_shape_t<E>,S> layout(user_shape);

  return detail::custom_bulk_execute_after(std::forward<E>(ex), std::forward<H>(before), layout.shape(), [=](auto native_coord)
  {
    std::invoke(user_function, layout[native_coord]);
  });
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

