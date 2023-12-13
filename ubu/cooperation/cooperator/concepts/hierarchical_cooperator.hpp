#pragma once

#include "../../../detail/prologue.hpp"

#include "../descend_with_group_coord.hpp"
#include "cooperator.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

template<class T>
concept hierarchical_cooperator =
  cooperator<T>
  and requires(T arg)
  {
    descend_with_group_coord(arg);
  }
;

template<hierarchical_cooperator C>
using child_cooperator_t = std::remove_cvref_t<decltype(get<0>(descend_with_group_coord(std::declval<C>())))>;

} // end ubu

#include "../../../detail/epilogue.hpp"

