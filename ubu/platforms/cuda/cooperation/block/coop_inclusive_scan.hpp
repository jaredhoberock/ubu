#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../cooperation/uninitialized_coop_optional_array.hpp"
#include "../../../../cooperation/cooperator/synchronize.hpp"
#include "../warp/coop_inclusive_scan.hpp"
#include "block_like.hpp"
#include <concepts>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{


template<block_like B, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr T coop_inclusive_scan(B group, T value, F binary_op)
{
  // each warp does a scan
  value = coop_inclusive_scan(subgroup(group), value, binary_op);

  // each thread computes its carry by summing previous warp sums
  uninitialized_coop_array<T,B> warp_sums(group, subgroup_count(group));

  // the final thread of each warp shares its sum
  if(is_last_in_group(subgroup(group)))
  {
    warp_sums[subgroup_id(group)] = value;
  }

  synchronize(group);

  T carry = warp_sums[0];

  for(int warp = 2; warp < subgroup_count(group); ++warp)
  {
    if(subgroup_id(group) >= warp)
    {
      carry = binary_op(warp_sums[warp-1], carry);
    }
  }

  // apply the carry to our thread's value
  if(subgroup_id(group) > 0)
  {
    value = binary_op(carry, value);
  }

  // XXX this not necessary if we put it in uninitialized_coop_array's dtor
  synchronize(group);

  return value;
}


template<ubu::cuda::block_like B, class T, std::invocable<T,T> F>
  requires (std::is_trivially_copy_constructible_v<T> and std::convertible_to<std::invoke_result_t<F,T,T>,T>)
constexpr std::optional<T> coop_inclusive_scan(B group, std::optional<T> value, F binary_op)
{
  // each warp does a scan
  value = coop_inclusive_scan(subgroup(group), value, binary_op);

  // XXX we actually don't need a subgroup sum for the final warp,
  //     but avoiding that extra allocation uses 2 more registers
  //     smem = num_warps * sizeof(T) + sizeof(std::uint32_t)
  //     21 registers
  uninitialized_coop_optional_array<T,B> warp_sums(group, subgroup_count(group));

  // the final thread of each warp shares its sum
  if(is_last_in_group(subgroup(group)))
  {
    warp_sums[subgroup_id(group)] = value;
  }

  synchronize(group);

  // each thread computes its carry by summing previous warp sums
  std::optional<T> carry;

  for(int warp = 1; warp < subgroup_count(group); ++warp)
  {
    if(subgroup_id(group) >= warp)
    {
      if(std::optional<T> other = warp_sums[warp-1]; other)
      {
        if(carry)
        {
          carry = binary_op(*other, *carry);
        }
        else
        {
          carry = other;
        }
      }
    }
  }

  // apply the carry to our thread's value
  if(carry)
  {
    if(value)
    {
      value = binary_op(*carry, *value);
    }
    else
    {
      value = carry;
    }
  }

  // XXX this is not necessary if we put it in uninitialized_coop_optional_array's dtor
  synchronize(group);

  return value;
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

