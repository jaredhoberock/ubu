#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperators/containers/uninitialized_coop_array.hpp"
#include "../warps/warp_like.hpp"
#include "block_like.hpp"
#include <optional>
#include <type_traits>

namespace ubu::cuda
{


template<cuda::block_like B, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr std::optional<T> coop_gather(B block, const std::optional<T>& value, int source)
{
  // we will gather up to one value per thread
  uninitialized_coop_array<T,B> values(block, size(block));

  // store my value if i have one
  if(value)
  {
    values[id(block)] = *value;
  }

  // each warp stores its bitmap
  uninitialized_coop_array<std::uint32_t,B> source_bitmaps(block, subgroup_count(block));

  int warp_bitmap = coop_ballot(subgroup(block), value.has_value());
  if(is_leader(subgroup(block)))
  {
    source_bitmaps[subgroup_id(block)] = warp_bitmap;
  }

  synchronize(block);

  // check if i received a value by examining the source bit
  int source_warp = source / cuda::warp_size;
  int source_lane = source % cuda::warp_size;
  bool received_value = source_bitmaps[source_warp] & (1 << source_lane);

  return received_value ? std::make_optional(values[source]) : std::nullopt;
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

