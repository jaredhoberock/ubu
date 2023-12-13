#pragma once

#include "../../../detail/prologue.hpp"

#include "../coop_alloca.hpp"
#include "cooperator.hpp"

namespace ubu
{

template<class T>
concept allocating_cooperator =
  cooperator<T>
  and requires(T self, int n)
  {
    coop_alloca(self, n); 
  }
;

} // end ubu

#include "../../../detail/epilogue.hpp"

