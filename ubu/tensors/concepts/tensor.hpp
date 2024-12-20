#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/traits/default_coordinate.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include <concepts>
#include <ranges>

namespace ubu
{
namespace detail
{

template<class T>
struct member_coordinate_or_default
{
  using type = default_coordinate_t<shape_t<T>>;
};

template<class T>
  requires requires { typename std::remove_cvref_t<T>::coordinate_type; }
struct member_coordinate_or_default<T>
{
  using type = typename std::remove_cvref_t<T>::coordinate_type;
};

// member_coordinate_or_default_t returns T::coordinate_type if it exists; otherwise, default_coordinate_t<shape_t<T>>
template<class T>
using member_coordinate_or_default_t = typename member_coordinate_or_default<T>::type;

} // end detail


template<class T>
concept tensor =
  // we must be able to get the shape of a tensor
  shaped<T>

  // the tensor's coordinate and shape types must be congruent
  and congruent<detail::member_coordinate_or_default_t<T>, shape_t<T>>

  // we must be able to access an element at a coordinate
  and requires(T t, detail::member_coordinate_or_default_t<T> coord)
  {
    // we must be able to get the tensor element at coord
    element(t, coord);

    // we must be able to check whether the tensor element at coord exists
    element_exists(t, coord);  
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

