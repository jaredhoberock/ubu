#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/integrals/ceil_div.hpp"
#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../primitives/concepts/allocating_cooperator.hpp"
#include "../primitives/concepts/hierarchical_cooperator.hpp"
#include "../primitives/subgroup.hpp"
#include "../primitives/subgroup_count.hpp"
#include "../primitives/subgroup_id.hpp"
#include "../primitives/subgroup_size.hpp"
#include "../primitives/synchronize.hpp"
#include "../uninitialized_coop_array.hpp"
#include <concepts>
#include <optional>

namespace ubu
{
namespace detail
{


template<allocating_cooperator C, class T, std::invocable<T,T> F>
  requires hierarchical_cooperator<C>
constexpr std::optional<T> coop_reduce_subgroups(C group, std::optional<T> value, F binary_op)
{
  // each subgroup does a reduction and sends the result to the subgroup leader
  value = coop_reduce(subgroup(group), value, binary_op);

  // at this point, only thread 0 of each subgroup can have a value

  // a shared communication buffer to gather to the 0th subgroup
  uninitialized_coop_array<T,C> buffer(group, subgroup_size(group));

  // each thread with a value will send it to a destination thread in subgroup 0
  integral_like auto dest_idx = subgroup_id(group);

  // each thread with a valid dest_idx and a value sends it to a thread in subgroup 0
  value = buffer.shuffle_if_valid_index(group, dest_idx, value);

  // XXX this loop can be omitted when C is a cuda::block_like, whether or not the block size is constant
  //     alternatively, maybe there is some way we could put a constant bound on cuda blocks' subgroup_size
  //     the ceil_div is like the parent/child subgroup branching factor ratio

  integral_like auto num_iterations = ceil_div(subgroup_count(group), subgroup_size(group));
  for(int iteration = 1; iteration < num_iterations; ++iteration)
  {
    dest_idx -= buffer.size();

    synchronize(group);

    // each thread with a valid dest_idx and a value sends it to a thread in subgroup 0
    if(std::optional other_value = buffer.shuffle_if_valid_index(group, dest_idx, value))
    {
      value = binary_op(*value, *other_value);
    }
  }

  return value;
}


} // end detail


template<allocating_cooperator C, class T, std::invocable<T,T> F>
  requires hierarchical_cooperator<C>
constexpr std::optional<T> coop_reduce(C self, std::optional<T> value, F binary_op)
{
  value = detail::coop_reduce_subgroups(self, value, binary_op);

  if(subgroup_id(self) == 0)
  {
    value = coop_reduce(subgroup(self), value, binary_op);
  }

  return value;
}


} // end ubu

#include "../../detail/epilogue.hpp"

