#pragma once

#include "../detail/prologue.hpp"

#include "grid.hpp"
#include "lattice.hpp"
#include "shape.hpp"

namespace ubu
{

// domain(g) returns a c++ Range of coordinates representing the domain of g
// XXX if G has a layout, then the order of the coordinates of the range
// returned really ought to be in "layout order"
// maybe another way to implement the result of domain would be to apply
// a lift function to indices in [0, size(g))
// XXX consider whether domain should be a customization point
template<grid G>
constexpr lattice<grid_shape_t<G>> domain(const G& g)
{
  return lattice(shape(g));
}

} // end ubu

#include "../detail/epilogue.hpp"

