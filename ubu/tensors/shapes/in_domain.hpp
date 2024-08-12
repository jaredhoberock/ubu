#pragma once

#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "shape.hpp"

namespace ubu
{

template<shaped T, coordinate_for<T> C>
constexpr bool in_domain(const T& tensor, const C& coord)
{
  if constexpr (sized<T> and integral_like<C>)
  {
    // treat sized, 1D tensors as a special case
    // in such a case, size may be smaller than shape
    // this establishes a tighter bound on 1D, sized domains
    return coord < size(tensor);
  }
  else
  {
    return is_below(coord, shape(tensor));
  }
}

} // end ubu

