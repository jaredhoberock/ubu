#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperators/containers/uninitialized_coop_array.hpp"
#include "../../../../cooperators/containers/uninitialized_coop_optional_array.hpp"
#include "../../../../cooperators/primitives.hpp"
#include "../warps/coop_inclusive_scan.hpp"
#include "../warps/shuffle_up.hpp"
#include "block_like.hpp"
#include <concepts>
#include <type_traits>

namespace ubu::cuda
{


template<block_like B, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr T coop_exclusive_scan(B group, T init, T value, F binary_op)
{
  // each warp does an inclusive scan
  value = coop_inclusive_scan(subgroup(group), value, binary_op);

  // shuffle to get the warp's exclusive scan (except for each lane 0)
  T exclusive_value = shuffle_up(subgroup(group), 1, value);

  // each thread computes its warp's init by summing previous warp sums
  uninitialized_coop_array<T,B> warp_sums(group, subgroup_count(group));

  if(is_last_in_group(subgroup(group)))
  {
    warp_sums[subgroup_id(group)] = value;
  }

  synchronize(group);

  // each thread accumulates previous warps' sums
  T carry_in = init;

  #pragma unroll
  for(int warp = 1; warp < subgroup_count(group); ++warp)
  {
    if(subgroup_id(group) >= warp)
    {
      carry_in = binary_op(carry_in, warp_sums[warp-1]);
    }
  }

  // accumulate the carry-in into our exclusive scan result,
  // except for each lane 0, whose result is simply its warp's carry-in
  return is_leader(subgroup(group)) ? carry_in : binary_op(carry_in, exclusive_value);
}


template<cuda::block_like B, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr std::optional<T> coop_exclusive_scan(B group, std::optional<T> init, std::optional<T> value, F binary_op)
{
  // each warp does an inclusive scan
  value = coop_inclusive_scan(subgroup(group), value, binary_op);

  // shuffle to get the warp's exclusive scan (except for each lane 0)
  std::optional exclusive_value = shuffle_up(subgroup(group), 1, value);

  // each thread computes its warp's init by summing previous warp sums
  uninitialized_coop_optional_array<T,B> warp_sums(group, subgroup_count(group));

  if(is_last_in_group(subgroup(group)))
  {
    warp_sums[subgroup_id(group)] = value;
  }

  synchronize(group);

  // each thread accumulates previous warps' sums
  std::optional carry_in = init;

  #pragma unroll
  for(int warp = 1; warp < subgroup_count(group); ++warp)
  {
    if(subgroup_id(group) >= warp)
    {
      if(std::optional<T> other = warp_sums[warp-1]; other)
      {
        if(carry_in)
        {
          carry_in = binary_op(*carry_in, *other);
        }
        else
        {
          carry_in = other;
        }
      }
    }
  }

  // accumulate the carry-in into our exclusive scan result,
  // except for each lane 0, whose result is simply its warp's carry-in

  if(is_leader(subgroup(group)))
  {
    exclusive_value = carry_in;
  }
  else if(carry_in)
  {
    if(exclusive_value)
    {
      exclusive_value = binary_op(*carry_in, *exclusive_value);
    }
    else
    {
      exclusive_value = carry_in;
    }
  }
  
  // XXX this is not necessary if uninitialized_coop_optional_array's dtor synchronizes
  synchronize(group);

  return exclusive_value;
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

