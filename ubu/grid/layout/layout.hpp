#pragma once

#include "../../detail/prologue.hpp"

#include "../grid.hpp"
#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"

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
  and coordinate_for<grid_element_t<L>, G>
;

} // end ubu

#include "../../detail/epilogue.hpp"

