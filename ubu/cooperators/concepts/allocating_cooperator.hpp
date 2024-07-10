#pragma once

#include "../../detail/prologue.hpp"

#include "../primitives/coop_alloca.hpp"
#include "../primitives/coop_dealloca.hpp"
#include "cooperator.hpp"

namespace ubu
{

template<class T>
concept allocating_cooperator =
  cooperator<T>
  and requires(T self, int n)
  {
    coop_alloca(self, n); 
    coop_dealloca(self, n);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

