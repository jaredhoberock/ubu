#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/rank.hpp"
#include "slicer.hpp"
#include <tuple>
#include <type_traits>

namespace ubu
{

template<class U, class C>
concept unslicer_for =
  slicer<U>
  and slicer<C>
  and (detail::underscore_count_v<U> == rank_v<C>)
;

} // end ubu

#include "../../detail/epilogue.hpp"

