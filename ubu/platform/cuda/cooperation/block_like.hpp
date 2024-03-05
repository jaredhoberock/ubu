#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../cooperation/cooperator/basic_cooperator.hpp"
#include "../../../cooperation/cooperator/concepts/cooperator.hpp"
#include "../../../cooperation/cooperator/traits/cooperator_thread_scope.hpp"
#include "../../../tensor/coordinate/traits/rank.hpp"
#include "../../../tensor/shape/shape.hpp"
#include "warp_like.hpp"
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>


namespace ubu::cuda
{


struct block_workspace
{
  // XXX we should use small_span or similar with int size
  std::span<std::byte> buffer;

  struct barrier_type
  {
    constexpr static const std::string_view thread_scope = "block";

    constexpr void arrive_and_wait() const
    {
#if defined(__CUDACC__)
      __syncthreads();
#endif
    }
  };

  barrier_type barrier;
};


template<class C>
concept block_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "block"
;


template<block_like B>
constexpr int synchronize_and_count(B, bool value)
{
#if defined(__CUDACC__)
  return __syncthreads_count(value);
#else
  return -1;
#endif
}


// overload descend_with_group_coord for 1D block_like groups
// this allows us to get a warp from a block which doesn't happen
// to have a hierarchical workspace
// returns the pair (warp_cooperator, which_warp)
template<block_like B>
  requires (rank_v<shape_t<B>> == 1)
constexpr auto descend_with_group_coord(B block)
{
  constexpr int warp_size = 32;
  int lane = coord(block) % warp_size;
  int which_warp = coord(block) / warp_size;
  return std::pair(basic_cooperator(lane, warp_size, warp_workspace{}), which_warp);
}


} // end ubu::cuda

#include "../../../detail/epilogue.hpp"

