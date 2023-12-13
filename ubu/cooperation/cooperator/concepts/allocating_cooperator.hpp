#pragma once

#include "../../../detail/prologue.hpp"

#include "../pop_shared_buffer.hpp"
#include "../push_shared_buffer.hpp"
#include "cooperator.hpp"

namespace ubu
{

template<class T>
concept allocating_cooperator =
  cooperator<T>
  and requires(T self, int n)
  {
    push_shared_buffer(self, n); 

    // XXX if cooperators must be passed to functions by value,
    //     then there's not really a reason to include pop_shared_buffer as
    //     an operation, because the cooperator stack variable will be cleared
    //     automatically
    //
    // XXX then, all we would really need is a single coop_alloca operation
    pop_shared_buffer(self, n);
  }
;

} // end ubu

#include "../../../detail/epilogue.hpp"

