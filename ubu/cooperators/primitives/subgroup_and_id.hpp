#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/integral_like.hpp"
#include "../concepts/hierarchical_cooperator.hpp"
#include "subgroup_and_coord.hpp"

namespace ubu
{

// this returns the pair (subgroup(group), subgroup_id)
// XXX eliminating this extra constraint would involve using a layout similar to ubu::id()
template<hierarchical_cooperator C>
  requires requires(C group)
  {
    { get<1>(subgroup_and_coord(group)) } -> integral_like;
  }
constexpr auto subgroup_and_id(C group)
{
  return subgroup_and_coord(group);
}

} // end ubu

#include "../../detail/epilogue.hpp"

