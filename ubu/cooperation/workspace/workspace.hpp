#pragma once

#include "../../detail/prologue.hpp"
#include "../../memory/buffer/get_buffer.hpp"
#include "../barrier/get_barrier.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

template<class T>
concept workspace = requires(T arg)
{
  get_buffer(arg);
  // XXX thread_scope_v<T> probably needs to be a requirement as well
  // XXX size(arg) seems like it could be useful but not actually a requirement
};

template<class T>
concept concurrent_workspace =
  workspace<T>
  and requires(T arg)
  {
    get_barrier(arg);
  }
;

template<concurrent_workspace W>
using workspace_barrier_t = barrier_t<W>;

} // end ubu

#include "../../detail/epilogue.hpp"

