#pragma once

#include "../../../detail/prologue.hpp"
#include "../bulk_execute_after.hpp"
#include <utility>

namespace ubu
{


template<class F, class E, class B, class S, class WS>
concept bulk_executable_on =
  requires(F func, E ex, B before, S shape, WS workspace_shape)
  {
    bulk_execute_after(std::forward<E>(ex), std::forward<B>(before), std::forward<S>(shape), std::forward<WS>(workspace_shape), std::forward<F>(func));
  }
;


} // end ubu

#include "../../../detail/epilogue.hpp"

