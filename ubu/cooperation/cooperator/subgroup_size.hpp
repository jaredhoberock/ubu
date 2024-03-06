#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/coordinate/concepts/integral_like.hpp"
#include "concepts/hierarchical_cooperator.hpp"
#include "size.hpp"
#include "subgroup.hpp"

namespace ubu
{

template<hierarchical_cooperator C>
constexpr integral_like auto subgroup_size(C group)
{
  return size(subgroup(group));
}

} // end ubu

#include "../../detail/epilogue.hpp"

