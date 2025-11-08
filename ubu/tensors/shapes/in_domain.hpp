#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/size.hpp"
#include "../coordinates/comparisons/is_inside.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/traits/rank.hpp"
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
    return is_strictly_inside(coord, shape(tensor));
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

