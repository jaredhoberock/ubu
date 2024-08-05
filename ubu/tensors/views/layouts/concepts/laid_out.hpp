#pragma once

#include "../../../../detail/prologue.hpp"

#include "../layout.hpp"

namespace ubu
{

// a type is "laid out" if it has a layout
// XXX the decomposable concept makes this concept superfluous
template<class T>
concept laid_out =
  requires(T arg)
  {
    layout(arg);
  }
;

} // end ubu

#include "../../../../detail/epilogue.hpp"

