#pragma once

#include "../../detail/prologue.hpp"

#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "concepts/integral_like.hpp"
#include "concepts/superdimensional.hpp"
#include "detail/tuple_algorithm.hpp"
#include "zeros.hpp"

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
    return detail::tuple_static_enumerate_similar_to<C>(ubu::zeros<R>, [&]<std::size_t index>(auto zero)
    {
      return truncate_coordinate<decltype(zero)>(get<index>(coord));
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

