#pragma once

#include "../../../detail/prologue.hpp"
#include "../new_bulk_execute_after.hpp"

namespace ubu
{


template<class F, class E, class B, class S, class WS>
concept bulk_executable_on =
  requires(F func, E ex, B before, S shape, WS workspace_shape)
  {
    new_bulk_execute_after(ex, before, shape, workspace_shape, func);
  }
;


} // end ubu

#include "../../../detail/epilogue.hpp"

