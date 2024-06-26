#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/comparisons/is_below.hpp"
#include "../concepts/tensor_like.hpp"
#include "../shapes.hpp"
#include "../traits/tensor_coordinate.hpp"
#include "../traits/tensor_shape.hpp"
#include "lattice.hpp"

namespace ubu
{

// domain(tensor) returns a view of coordinates representing the domain of tensor
// XXX if T has a layout, then the order of the coordinates of the range
// returned really ought to be in "layout order"
// maybe another way to implement the result of domain would be to apply
// a lift function to indices in [0, size(tensor))
// XXX consider whether domain should be a customization point
template<tensor_like T>
constexpr lattice<tensor_coordinate_t<T>,tensor_shape_t<T>> domain(const T& tensor)
{
  return lattice<tensor_coordinate_t<T>,tensor_shape_t<T>>(shape(tensor));
}

// returns true if coord[i] is < shape(tensor)[i] for all i in rank_v<C>
template<tensor_like T, coordinate_for<T> C>
constexpr bool in_domain(const T& tensor, const C& coord)
{
  return is_below(coord, shape(tensor));
}

} // end ubu

#include "../../detail/epilogue.hpp"

