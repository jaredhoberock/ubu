#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant_valued.hpp"
#include "../../miscellaneous/integrals/bounded.hpp"
#include "concepts/congruent.hpp"
#include "concepts/coordinate.hpp"
#include "detail/tuple_algorithm.hpp"

#include <limits>

namespace ubu
{


// attempts to introduce a static upper bound on a coordinate
// if upper_bound is not constant_valued, then this function is the identity
// XXX maybe this should be named try_bound_coordinate
template<coordinate C, congruent<C> U>
constexpr congruent<C> auto bound_coordinate(const C& coord, const U& upper_bound)
{
  if constexpr (integral_like<C>)
  {
    // if upper_bound is a constant and coord is not
    if constexpr (constant_valued<U> and (not constant_valued<C>))
    {
      // and if the value of upper_bound is smaller than the maximum possible value of coordinate
      if constexpr (U{} < std::numeric_limits<C>::max())
      {
        // then upper_bound is a tighter bound on coord

        // convert upper_bound to coord's integral type
        using integral_type = to_integral_t<C>;
        constexpr integral_type b = to_integral(upper_bound);

        return bounded<b>(coord);
      }
      else
      {
        return coord;
      }
    }
    else
    {
      return coord;
    }
  }
  else
  {
    // coord and upper_bound are tuples, recurse
    return detail::tuple_zip_with(coord, upper_bound, [](auto c, auto ub)
    {
      return bound_coordinate(c,ub);
    });
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

