#pragma once

#include "../../detail/prologue.hpp"
#include "pop_allocation.hpp"
#include "push_allocation.hpp"
#include "workspace.hpp"

namespace ubu
{

template<class T>
concept stack_workspace =
  workspace<T>
  and requires(T& ws, int n)
  {
    pop_allocation(ws, n);
    { push_allocation(ws, n) } -> pointer_like;
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

