#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../../miscellaneous/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "concepts/superdimensional.hpp"
#include "traits/zeros.hpp"

namespace ubu
{


template<coordinate R, superdimensional<R> C>
constexpr congruent<R> auto truncate_coordinate(const C& coord)
{
  if constexpr (scalar_coordinate<R>)
  {
    if constexpr (integral_like<C>)
    {
      return coord;
    }
    else
    {
      return truncate_coordinate<R>(get<0>(coord));
    }
  }
  else
  {
    return tuples::static_enumerate_similar_to<C>(zeros_v<R>, [&]<std::size_t index>(auto zero)
    {
      return truncate_coordinate<decltype(zero)>(get<index>(coord));
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

