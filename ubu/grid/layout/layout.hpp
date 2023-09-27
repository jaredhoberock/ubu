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

template<class L, class Coord>
concept layout_for =
  layout<L>
  and coordinate<Coord>
  and requires(L layout, Coord coord)
  {
    { layout[coord] } -> coordinate;
  }
;

template<class L, class FromCoord, class ToCoord>
concept layout_onto =
  layout_for<L, FromCoord>
  and coordinate<ToCoord>
  and requires(L layout, FromCoord coord)
  {
    { layout[coord] } -> congruent<ToCoord>;
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

