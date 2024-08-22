#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/size.hpp"
#include "../coordinates/comparisons/is_below.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/traits/rank.hpp"
#include "shape.hpp"

namespace ubu
{

template<shaped T, coordinate_for<T> C>
constexpr bool in_domain(const T& tensor, const C& coord)
{
  if constexpr (rank_v<C> == 0)
  {
    // a rank-0 coord for a rank-0 tensor is defined to be in its domain
    return true;
  }
  else if constexpr (sized<T> and integral_like<C>)
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

#include "../../detail/epilogue.hpp"

