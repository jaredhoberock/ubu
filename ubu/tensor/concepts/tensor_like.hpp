#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../coordinate/zeros.hpp"
#include "../element_exists.hpp"
#include "../shape/shape.hpp"
#include <concepts>
#include <ranges>

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
concept tensor_like =
  // we must be able to get the shape of a tensor
  requires(T g)
  {
    shape(g);
  }

  // the tensor's coordinate and shape types must be congruent
  and congruent<detail::coordinate_or_default_t<T>, shape_t<T>>

  // we must be able to access an element at a coordinate
  and requires(T g, detail::coordinate_or_default_t<T> c)
  {
    element(g, c);         // we must be able to get the tensor element at c
    element_exists(g, c);  // we must be able to check whether the tensor element at c exists
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"
