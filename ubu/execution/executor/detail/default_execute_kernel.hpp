#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../tensor/coordinate/compare.hpp"
#include "../../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../../tensor/coordinate/zeros.hpp"
#include "../../../tensor/layout/layout.hpp"
#include "../../../tensor/shape.hpp"
#include "../bulk_execute.hpp"
#include "../concepts/executor.hpp"
#include "../kernel_layout.hpp"
#include "../traits/executor_workspace.hpp"
#include "../traits/executor_workspace_shape.hpp"
#include <concepts>

namespace ubu::detail
{


// when execute_kernel can map directly onto bulk_execute without a layout
// i.e., f is an invocable of the executor's coordinate type
template<executor E, coordinate S, std::invocable<executor_coordinate_t<E>> F>
void default_execute_kernel(E ex, const S& shape, F&& f)
{
  bulk_execute(ex, shape, std::forward<F>(f));
}


// when execute_kernel cannot map only bulk_execute without a layout
// i.e., f is not an invocable of the executor's coordinate type
template<executor E, coordinate S, std::invocable<S> F>
  requires (not std::invocable<F, executor_coordinate_t<E>>)
void default_execute_kernel(E ex, const S& user_shape, F&& f)
{
  // to_user_coord is a layout which maps a coordinate originating from
  // the executor to a coordinate within the user's requested shape
  layout auto to_user_coord = kernel_layout(ex, user_shape, std::forward<F>(f));

  bulk_execute(ex, shape(to_user_coord), [=](const executor_coordinate_t<E>& ex_coord)
  {
    S user_coord = to_user_coord[ex_coord];
    if(is_below(user_coord, user_shape))
    {
      f(user_coord);
    }
  });
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

