#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/integrals/integral_like.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "concepts/subdimensional.hpp"
#include "concepts/weakly_congruent.hpp"
#include "detail/tuple_algorithm.hpp"
#include "traits/ones.hpp"

namespace ubu
{

// returns a coordinate congruent to R by one-extending the corresponding modes that are missing from coord
// postcondition: shape_size(result) == shape_size(shape)
template<coordinate R, subdimensional<R> C>
constexpr congruent<R> auto one_extend_coordinate(const C& coord)
{
  // XXX we should probably use a solution that would yield
  //     constant<1> as elements of the result of this function
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
    static_assert(detail::tuple_like<R>);

    // create a replacement for the leftmost element of ones_v<R> by recursing on that element
    auto replacement = one_extend_coordinate<std::tuple_element_t<0,R>>(coord);

    // take ones_v<R> and replace element 0 with our replacement
    return detail::tuple_replace_element<0>(ones_v<R>, replacement);
  }
  else
  {
    // recursive case 2: both R and C are tuples
    static_assert(detail::tuple_like<R>);
    static_assert(detail::tuple_like<C>);

    // map one_extend_coordinate over the elements of ones_v<R>
    return detail::tuple_static_enumerate_similar_to<C>(ones_v<R>, [&]<std::size_t index>(auto one_i)
    {
      if constexpr (index < rank_v<C>)
      {
        // this element's index is in coord, so recurse down that element of coord
        return one_extend_coordinate<decltype(one_i)>(get<index>(coord));
      }
      else
      {
        // there's no corresponding element of coord, so this element becomes a one
        return one_i;
      }
    });
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

