#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperation/uninitialized_coop_array.hpp"
#include "../warp/warp_like.hpp"
#include "block_like.hpp"
#include <atomic>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{


// XXX this implementation isn't particularly specific to CUDA blocks,
//     but it does assume that the subgroup size is 32 (i.e., warp_size)
// XXX and it should actually require that the subgroup_size is 32
template<block_like B, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr std::optional<T> coop_scatter(B block, const std::optional<T>& value, int destination)
{
  // we will store one bitmap per warp
  uninitialized_coop_array<std::uint32_t,B> destination_bitmaps(block, subgroup_count(block));
  
  // we will scatter up to one value per thread
  uninitialized_coop_array<T,B> values(block, size(block));

  // decompose the destination into a warp and lane
  int destination_warp = destination / cuda::warp_size;
  int destination_lane = destination % cuda::warp_size;

  if(std::atomic_ref destination_bitmap(destination_bitmaps[destination_warp]); value)
  {
    // set the destination bit and store the value
    destination_bitmap.fetch_or(1 << destination_lane);
    values[destination] = *value;
  }
  else
  {
    // clear the destination bit
    destination_bitmap.fetch_and(~(1 << destination_lane));
  }

  synchronize(block);

  std::optional<T> result;

  // check if i received a value by examining my bit
  int my_lane = id(subgroup(block));
  bool received_value = destination_bitmaps[subgroup_id(block)] & (1 << my_lane);
  if(received_value)
  {
    result = values[id(block)];
  }

  // XXX think we need a synchronize here

  return result;
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

