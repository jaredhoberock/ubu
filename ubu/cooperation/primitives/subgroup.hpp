#pragma once

#include "../../detail/prologue.hpp"

#include "concepts/hierarchical_cooperator.hpp"
#include "subgroup_and_coord.hpp"

namespace ubu
{

template<hierarchical_cooperator C>
constexpr cooperator auto subgroup(const C& self)
{
  return get<0>(subgroup_and_coord(self));
}

} // end ubu

#include "../../detail/epilogue.hpp"
