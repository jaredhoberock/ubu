#pragma once

#include "../../../../detail/prologue.hpp"
#include "../bulk_execute_after.hpp"
#include <utility>

namespace ubu
{


template<class F, class E, class B, class S>
concept bulk_executable_on =
  requires(F func, E ex, B before, S shape)
  {
    bulk_execute_after(std::forward<E>(ex), std::forward<B>(before), std::forward<S>(shape), std::forward<F>(func));
  }
;


} // end ubu

#include "../../../../detail/epilogue.hpp"

