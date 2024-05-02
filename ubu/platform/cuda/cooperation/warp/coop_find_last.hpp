#pragma once

#include "../../../../detail/prologue.hpp"
#include "coop_ballot.hpp"
#include "coop_reduce.hpp"
#include "warp_like.hpp"

namespace ubu::cuda
{

// if any threads' predicate is true, returns the greatest of their ids
// otherwise, if no predicates are true, returns -1
// XXX might be better to return size(warp) otherwise
//     the idea of returning -1 is if we were doing this sequentially,
//     we'd start at the end of the list and work our way down looking
//     for a true predicate
template<warp_like W>
constexpr int coop_find_last(W warp, bool predicate) noexcept
{
#if defined(__CUDACC__)
  unsigned int ballot = coop_ballot(warp, predicate); 
  if(ballot == 0) return -1;
  return 31 - __clz(ballot);
#else
  return -1;
#endif
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

