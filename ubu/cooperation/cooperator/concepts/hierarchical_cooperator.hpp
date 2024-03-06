#pragma once

#include "../../../detail/prologue.hpp"

#include "../subgroup_and_coord.hpp"
#include "semicooperator.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

template<class T>
concept hierarchical_cooperator =
  semicooperator<T>
  and requires(T arg)
  {
    subgroup_and_coord(arg);
  }
;

template<hierarchical_cooperator C>
using child_cooperator_t = std::remove_cvref_t<decltype(get<0>(subgroup_and_coord(std::declval<C>())))>;


} // end ubu

#include "../../../detail/epilogue.hpp"

