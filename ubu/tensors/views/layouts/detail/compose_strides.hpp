#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../utilities/tuples.hpp"
#include "../../../coordinates/concepts/congruent.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/concepts/equal_rank.hpp"
#include "../../../coordinates/detail/to_integral_like.hpp"


namespace ubu::detail
{


template<coordinate A, coordinate B>
  requires (rank_v<A> != 0 and rank_v<A> <= rank_v<A>)
constexpr congruent<B> auto compose_strides(const A& a, const B& b)
{
  if constexpr (unary_coordinate<A> and unary_coordinate<B>)
  {
    return to_integral_like(a) * to_integral_like(b);
  }
  else if constexpr (rank_v<A> < rank_v<B>)
  {
    return tuples::zip_with(b, [&](const auto& bi)
    {
      return compose_strides(a, bi);
    });
  }
  else
  {
    static_assert(equal_rank<A,B>);

    return tuples::zip_with(a, b, [](const auto& ai, const auto& bi)
    {
      return compose_strides(ai,bi);
    });
  }
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

