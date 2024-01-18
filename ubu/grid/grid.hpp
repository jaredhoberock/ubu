#pragma once

#include "../detail/prologue.hpp"
#include "coordinate/concepts/coordinate.hpp"
#include "coordinate/zeros.hpp"
#include "element_exists.hpp"
#include "shape/shape.hpp"
#include <concepts>
#include <ranges>
#include <type_traits>

namespace ubu
{
namespace detail
{

template<class T>
struct coordinate_or_default
{
  using type = zeros_t<shape_t<T>>;
};

template<class T>
  requires requires { typename std::remove_cvref_t<T>::coordinate_type; }
struct coordinate_or_default<T>
{
  using type = typename std::remove_cvref_t<T>::coordinate_type;
};

// coordinate_or_default_t returns T::coordinate_type if it exists; otherwise, zeros_t<shape_t<T>>
template<class T>
using coordinate_or_default_t = typename coordinate_or_default<T>::type;

} // end detail


template<class T>
concept grid =
  // we must be able to get the shape of a grid
  requires(T g)
  {
    shape(g);
  }

  // the grid's coordinate and shape types must be congruent
  and congruent<detail::coordinate_or_default_t<T>, shape_t<T>>

  // we must be able to access the grid element at a coordinate
  and requires(T g, detail::coordinate_or_default_t<T> c)
  {
    element(g, c);         // we must be able to get the grid element at c
    element_exists(g, c);  // we must be able to check whether the grid element at c exists
  }
;

template<grid T>
using grid_shape_t = shape_t<T>;

template<grid T>
using grid_coordinate_t = detail::coordinate_or_default_t<T>;

template<grid T>
using grid_reference_t = decltype(std::declval<T>()[std::declval<grid_coordinate_t<T>>()]);

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

