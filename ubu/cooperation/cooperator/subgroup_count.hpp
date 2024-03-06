#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/coordinate/concepts/integral_like.hpp"
#include "concepts/hierarchical_cooperator.hpp"
#include "size.hpp"
#include "subgroup.hpp"

namespace ubu
{

template<hierarchical_cooperator C>
constexpr integral_like auto subgroup_count(C group)
{
  return size(group) / size(subgroup(group));
}


} // end ubu

#include "../../detail/epilogue.hpp"

