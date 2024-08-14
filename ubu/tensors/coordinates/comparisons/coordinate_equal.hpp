#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"

namespace ubu
{

template<ubu::coordinate A, ubu::congruent<A> B>
constexpr bool coordinate_equal(const A& a, const B& b)
{
  if constexpr (rank_v<A> == 1)
  {
    return detail::to_integral_like(a) == detail::to_integral_like(b);
  }
  else
  {
    return tuples::equal(a, b, [](auto a_i, auto b_i)
    {
      return coordinate_equal(a_i, b_i);
    });
  }
}

} // end ubu

#include "../../../detail/epilogue.hpp"

