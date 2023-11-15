#pragma once

#include "../detail/prologue.hpp"
#include "coordinate/coordinate.hpp"
#include "shape/shape.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{
namespace detail
{

template<class T>
concept not_void = not std::is_void_v<T>;

} // end detail

template<class T, class I>
concept indexable_by =
  requires(T obj, I idx)
  {
    // XXX we could base this on element(obj,idx) instead of bracket
    { obj[idx] } -> detail::not_void;
  }
;

template<class T>
concept grid =
  requires(T g)
  {
    shape(g);
  }
  and indexable_by<T, shape_t<T>>
;

// XXX it might be more consistent with the allocator/executor stuff if we named this
//     grid_coordinate_t instead of grid_shape_t, especially considering we already have the name shape_t
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

} // end ubu

#include "../detail/epilogue.hpp"

