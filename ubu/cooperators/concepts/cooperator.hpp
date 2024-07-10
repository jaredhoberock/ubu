#pragma once

#include "../../detail/prologue.hpp"

#include "../primitives/synchronize.hpp"
#include "semicooperator.hpp"

namespace ubu
{

template<class T>
concept cooperator =
  semicooperator<T>
  and requires(T arg)
  {
    synchronize(arg);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

