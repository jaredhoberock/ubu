#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/integral_like.hpp"
#include "../concepts/hierarchical_cooperator.hpp"
#include "subgroup_and_id.hpp"

namespace ubu
{

template<hierarchical_cooperator C>
constexpr integral_like auto subgroup_id(C group)
{
  return get<1>(subgroup_and_id(group));
}

} // end ubu

#include "../../detail/epilogue.hpp"

