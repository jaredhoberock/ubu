#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../../tensors/coordinates/concepts/coordinate.hpp"
#include "../../../../tensors/coordinates/concepts/weakly_congruent.hpp"
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

struct bulk_invocable_archetype
{
  template<class C>
  void operator()(C coord) const;
};

} // end detail

template<class E>
concept bulk_executor =
  executor<E>
  and congruent<executor_shape_t<E>, executor_coordinate_t<E>>
  and bulk_executable_on<detail::bulk_invocable_archetype, E, executor_happening_t<E>, executor_shape_t<E>>
  and weakly_congruent<executor_workspace_shape_t<E>, executor_coordinate_t<E>>
;

} // end ubu

#include "../../../../detail/epilogue.hpp"

