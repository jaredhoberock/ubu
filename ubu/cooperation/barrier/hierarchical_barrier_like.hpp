#pragma once

#include "../../detail/prologue.hpp"

#include "barrier_like.hpp"
#include "get_local_barrier.hpp"

namespace ubu
{

template<class T>
concept hierarchical_barrier_like =
  barrier_like<T>
  and requires(T arg)
  {
    get_local_barrier(arg);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

