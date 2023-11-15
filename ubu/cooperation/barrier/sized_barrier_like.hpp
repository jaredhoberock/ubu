#pragma once

#include "../../detail/prologue.hpp"

#include "barrier_like.hpp"
#include <ranges>

namespace ubu
{

template<class T>
concept sized_barrier_like =
  barrier_like<T>
  and requires(T arg)
  {
    std::ranges::size(arg);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

