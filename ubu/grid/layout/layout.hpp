#pragma once

#include "../../detail/prologue.hpp"

#include "../grid.hpp"
#include "../coordinate/congruent.hpp"
#include "../coordinate/coordinate.hpp"

namespace ubu
{

template<class T>
concept layout =
  grid<T>
  and coordinate<grid_element_t<T>>
;

template<class L, class G>
concept layout_for =
  layout<L>
  and indexable_by<G, grid_element_t<L>>
;

} // end ubu

#include "../../detail/epilogue.hpp"

