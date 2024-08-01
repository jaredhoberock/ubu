#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/integrals/integral_like.hpp"
#include "../../utilities/tuples.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "concepts/subdimensional.hpp"
#include "concepts/weakly_congruent.hpp"
#include "traits/zeros.hpp"

namespace ubu
{

// returns a coordinate congruent to R by zero-extending the corresponding modes that are missing from coord
// postcondition: shape_size(result) == shape_size(shape)
template<coordinate R, subdimensional<R> C>
constexpr congruent<R> auto zero_extend_coordinate(const C& coord)
{
  // XXX we should probably use a solution that would yield
  //     constant<0> as elements of the result of this function
  //
  //     or we just need a generalized extend_coordinate<R>(coord, integral_like)

  if constexpr (congruent<R,C>)
  {
    // terminal case: R and C are congruent, there's nothing to extend
    return coord;
  }
  else if constexpr (integral_like<C>)
  {
    // recursive case 1: R is a tuple and C is integral
    static_assert(tuples::tuple_like<R>);

    // create a replacement for the leftmost element of zeros_v<R> by recursing on that element
    auto replacement = zero_extend_coordinate<tuples::first_t<R>>(coord);

    // take zeros_v<R> and replace element 0 with our replacement
    return tuples::replace_element<0>(zeros_v<R>, replacement);
  }
  else
  {
    // recursive case 2: both R and C are tuples
    static_assert(tuples::tuple_like<R>);
    static_assert(tuples::tuple_like<C>);

    // map one_extend_coordinate over the elements of zeros_v<R>
    return tuples::static_enumerate_like<C>(zeros_v<R>, [&]<std::size_t index>(auto zero_i)
    {
      if constexpr (index < rank_v<C>)
      {
        // this element's index is in coord, so recurse down that element of coord
        return zero_extend_coordinate<decltype(zero_i)>(get<index>(coord));
      }
      else
      {
        // there's no corresponding element of coord, so this element becomes a zero
        return zero_i;
      }
    });
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

