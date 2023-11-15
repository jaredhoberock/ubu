#pragma once

#include "../../detail/prologue.hpp"

#include "descend_with_group_coord.hpp"
#include "hierarchical_cooperator.hpp"

namespace ubu
{

template<hierarchical_cooperator C>
constexpr cooperator auto descend(const C& self)
{
  return get<0>(descend_with_group_coord(self));
}

} // end ubu

#include "../../detail/epilogue.hpp"
