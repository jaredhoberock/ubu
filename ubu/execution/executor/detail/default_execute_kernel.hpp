#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../grid/coordinate/compare.hpp"
#include "../../../grid/coordinate/coordinate.hpp"
#include "../../../grid/coordinate/zeros.hpp"
#include "../../../grid/layout/layout.hpp"
#include "../../../grid/shape.hpp"
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
  // we don't need a workspace
  auto workspace_shape = zeros<executor_workspace_shape_t<E>>;

  bulk_execute(ex, shape, workspace_shape, [=](const executor_coordinate_t<E>& coord, executor_workspace_t<E>)
  {
    f(coord);
  });
}


// when execute_kernel cannot map only bulk_execute without a layout
// i.e., f is not an invocable of the executor's coordinate type
template<executor E, coordinate S, std::invocable<S> F>
  requires (not std::invocable<F, executor_coordinate_t<E>>)
void default_execute_kernel(E ex, const S& user_shape, F&& f)
{
  // we don't need a workspace
  // XXX why is this producing size_t?
  auto workspace_shape = zeros<executor_workspace_shape_t<E>>;

  // to_user_coord is a layout which maps a coordinate originating from
  // the executor to a coordinate within the user's requested shape
  layout auto to_user_coord = kernel_layout(ex, user_shape, std::forward<F>(f));

  bulk_execute(ex, shape(to_user_coord), workspace_shape, [=](const executor_coordinate_t<E>& ex_coord, executor_workspace_t<E>)
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

