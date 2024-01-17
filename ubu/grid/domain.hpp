#pragma once

#include "../detail/prologue.hpp"

#include "coordinate/compare/is_below.hpp"
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
constexpr lattice<grid_coordinate_t<G>,grid_shape_t<G>> domain(const G& g)
{
  return lattice(shape(g));
}

// returns true if coord[i] is < shape(grid)[i] for all i in rank_v<C>
template<grid G, coordinate_for<G> C>
constexpr bool in_domain(const G& grid, const C& coord)
{
  return is_below(coord, shape(grid));
}

} // end ubu

#include "../detail/epilogue.hpp"

