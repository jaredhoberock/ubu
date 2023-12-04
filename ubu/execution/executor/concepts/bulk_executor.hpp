#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../grid/coordinate/congruent.hpp"
#include "../../../grid/coordinate/coordinate.hpp"
#include "../../../grid/coordinate/weakly_congruent.hpp"
#include "../traits/executor_coordinate.hpp"
#include "../traits/executor_happening.hpp"
#include "../traits/executor_shape.hpp"
#include "../traits/executor_workspace_shape.hpp"
#include "bulk_executable_on.hpp"
#include "executor.hpp"


namespace ubu
{
namespace detail
{

struct new_bulk_invocable_archetype
{
  template<class C, class W>
  void operator()(C coord, W workspace) const;
};

} // end detail

template<class E>
concept bulk_executor =
  executor<E>
  and congruent<executor_shape_t<E>, executor_coordinate_t<E>>
  and weakly_congruent<executor_workspace_shape_t<E>, executor_coordinate_t<E>>
  and bulk_executable_on<detail::new_bulk_invocable_archetype, E, executor_happening_t<E>, executor_shape_t<E>, executor_workspace_shape_t<E>>
;

} // end ubu

#include "../../../detail/epilogue.hpp"

