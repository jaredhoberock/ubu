#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"
#include "../traits/rank.hpp"


namespace ubu
{


// is_below(lhs, rhs) returns true if all modes of lhs is < their corresponding mode of rhs

template<coordinate A, coordinate B>
  requires (congruent<A,B> and rank_v<A> > 0)
constexpr bool is_below(const A& a, const B& b)
{
  if constexpr (rank_v<A> == 1)
  {
    return detail::to_integral_like(a) < detail::to_integral_like(b);
  }
  else
  {
    auto tuple_of_results = tuples::zip_with(a, b, [](auto a_i, auto b_i)
    {
      return is_below(a_i, b_i);
    });

    return tuples::all_of(tuple_of_results, [](bool result)
    {
      return result;
    });
  }
}


// is_below_or_equal(lhs, rhs) returns true if all modes of lhs is <= their corresponding mode of rhs

template<coordinate A, coordinate B>
  requires (congruent<A,B> and rank_v<A> > 0)
constexpr bool is_below_or_equal(const A& a, const B& b)
{
  if constexpr (rank_v<A> == 1)
  {
    return detail::to_integral_like(a) <= detail::to_integral_like(b);
  }
  else
  {
    auto tuple_of_results = tuples::zip_with(a, b, [](auto a_i, auto b_i)
    {
      return is_below_or_equal(a_i, b_i);
    });

    return tuples::all_of(tuple_of_results, [](bool result)
    {
      return result;
    });
  }
}


} // end ubu

#include "../../../detail/epilogue.hpp"

