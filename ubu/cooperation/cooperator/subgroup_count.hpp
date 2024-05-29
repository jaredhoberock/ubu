#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integral/integral_like.hpp"
#include "concepts/hierarchical_cooperator.hpp"
#include "size.hpp"
#include "subgroup_size.hpp"

namespace ubu
{

template<hierarchical_cooperator C>
constexpr integral_like auto subgroup_count(C group)
{
  return size(group) / subgroup_size(group);
}


} // end ubu

#include "../../detail/epilogue.hpp"

