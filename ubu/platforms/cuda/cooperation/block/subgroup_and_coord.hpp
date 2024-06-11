#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperation/cooperators/basic_cooperator.hpp"
#include "../../../../tensors/coordinates/traits/rank.hpp"
#include "../../../../tensors/shapes/shape.hpp"
#include "../warp/warp_like.hpp"
#include "block_like.hpp"
#include <utility>

namespace ubu::cuda
{

// overload subgroup_and_coord for 1D block_like groups
// this allows us to get a warp from a block which doesn't happen
// to have a hierarchical workspace
// returns the pair (warp_cooperator, which_warp)
template<block_like B>
  requires (rank_v<shape_t<B>> == 1)
constexpr auto subgroup_and_coord(B block)
{
  int lane = coord(block) % warp_size;
  int which_warp = coord(block) / warp_size;
  return std::pair(basic_cooperator(lane, warp_size, warp_workspace{}), which_warp);
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

