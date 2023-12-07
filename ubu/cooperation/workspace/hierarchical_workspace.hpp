#pragma once

#include "../../detail/prologue.hpp"

#include "get_local_workspace.hpp"
#include "workspace.hpp"

namespace ubu
{

template<class T>
concept hierarchical_workspace =
  workspace<T>
  and requires(T arg)
  {
    get_local_workspace(arg);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

