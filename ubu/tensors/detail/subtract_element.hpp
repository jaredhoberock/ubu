#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/detail/as_integral_like.hpp"
#include "../coordinates/detail/tuple_algorithm.hpp"
#include "../coordinates/traits/rank.hpp"

namespace ubu::detail
{

// subtracts b from a[i] and returns a result congruent to a
template<std::size_t i, semicoordinate A, coordinate B>
  requires ((i < rank_v<A>)                              // a[i] must exist
            and coordinate<coordinate_element_t<i,A>>    // a[i] must be a coordinate
            and congruent<coordinate_element_t<i,A>, B>) // a[i] must be congruent to b
constexpr congruent<A> auto subtract_element(const A& a, const B& b)
{
  if constexpr (scalar_coordinate<A>)
  {
    return as_integral_like(a) - as_integral_like(b);
  }
  else
  {
    // auto indices = tuple(constant<I>()...);
    constexpr auto indices = as_tuple_of_constants(tuple_indices<A>);

    return tuple_zip_with(a, indices, [&](const auto& a_i, auto index)
    {
      if constexpr (index == i)
      {
        return coordinate_difference(a_i, b);
      }
      else
      {
        return a_i;
      }
    });
  }
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

