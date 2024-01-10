#pragma once

#include "../detail/prologue.hpp"
#include "coordinate/concepts/coordinate.hpp"
#include "element_exists.hpp"
#include "shape/shape.hpp"
#include <concepts>
#include <ranges>
#include <type_traits>

namespace ubu
{

template<class T>
concept grid =
  requires(T g)
  {
    shape(g);
  }
  and coordinate_for<shape_t<T>, T>
  and requires(T g, shape_t<T> c)
  {
    element_exists(g, c);
  }
;

// XXX in addition to grid_shape_t, I think we also need grid_coordinate_t
//     in some important cases, the coordinate type will differ from the shape type
//     for example, grids whose shape is known to be a constant at compile time
//     would have a grid_shape_t like std::integral_constant, while their coordinate
//     type would still be able to vary dynamically

template<grid T>
using grid_shape_t = shape_t<T>;

template<grid T>
using grid_reference_t = decltype(std::declval<T>()[std::declval<grid_shape_t<T>>()]);

template<grid T>
using grid_element_t = std::remove_cvref_t<grid_reference_t<T>>;

template<class G, class T>
concept grid_of =
  grid<G>
  and std::same_as<grid_element_t<G>,T>
;

template<class T>
concept sized_grid =
  grid<T>
  and requires(T g)
  {
    std::ranges::size(g);
  }
;

} // end ubu

#include "../detail/epilogue.hpp"

