#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/constant.hpp"
#include "../../../utilities/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "to_integral_like.hpp"
#include <utility>

namespace ubu::detail
{


// coordinate_inclusive_scan is a recursive inclusive scan operation on a coordinate
// The carry_in parameter is also a coordinate which is weakly_congruent with coord
//
// The result of this function is a pair of values (result_tuple, carry_out):
//   1. result_tuple is a tuple whose size is the same rank as the input coordinate, and
//   2. carry_out is an integer, and is the carry_out result of the final application
//      of the combine function
//
// When the result of the combine function is an integer, then the result_tuple is a
// coordinate conguent with C1.
//
// The combination function has control over the value of the carry after each combination
// combine(carry_in, input) must return the pair (result, carry_out)
//
// The types of combine's input and carry_in parameters must be integers.
// The type of combine's carry_out result must be an integer.

template<coordinate C1, weakly_congruent<C1> C2, class F>
constexpr tuples::pair_like auto coordinate_inclusive_scan_and_fold(const C1& coord, const C2& carry_in, const F& combine)
{
  if constexpr (nullary_coordinate<C1> and nullary_coordinate<C2>)
  {
    // terminal case 0: coord and carry_in are both (), return the pair ((), ())
    return std::pair(coord, carry_in);
  }
  else if constexpr (unary_coordinate<C1>)
  {
    // terminal case 1: coord is unary, carry_in is either unary or nullary; call combine
    // if carry_in is (), we map that to zero
    if constexpr (nullary_coordinate<C2>)
    {
      return combine(0_c, detail::to_integral_like(coord));
    }
    else
    {
      return combine(detail::to_integral_like(carry_in), detail::to_integral_like(coord));
    }
  }
  else if constexpr (equal_rank<C1, C2>)
  {
    // recursive case 0: coord and carry_in are non-empty tuples of the same rank
    // map coordinate_inclusive_scan_and_fold across coord & carry_in and unzip the result
    return tuples::unzip(tuples::zip_with(coord, carry_in, [&](auto coord_i, auto carry_i)
    {
      return coordinate_inclusive_scan_and_fold(coord_i, carry_i, combine);
    }));
  }
  else
  {
    // recursive case 1: coord is multiary and carry_in is either nullary or unary
    // recursively scan and fold across coord using coordinate_inclusive_scan_and_fold as the fold operation
    static_assert(multiary_coordinate<C1>);
    static_assert(nullary_coordinate<C2> or unary_coordinate<C2>);

    return tuples::inclusive_scan_and_fold(coord, carry_in, [&](auto carry_i, auto coord_i)
    {
      return coordinate_inclusive_scan_and_fold(coord_i, carry_i, combine);
    });
  }
}


// coordinate_inclusive_scan_with_final is like coordinate_inclusive_scan except that it treats the final combination operation in a special way
// it yields a single result which is a tuple the same size as the input coordinate's rank
//
// The final combination function has control over the result of this function:
// 
//     final_combine(integral_like final_carry_in, integral_like final_input) -> result
//
// When final_combine's result is integral_like, then the result of coordinate_inclusive_scan_with_final is congruent<C1>
template<coordinate C1, weakly_congruent<C1> C2, class F1, class F2>
constexpr auto coordinate_inclusive_scan_with_final(const C1& coord, const C2& carry_in, const F1& combine, const F2& final_combine)
{
  if constexpr (unary_coordinate<C1>)
  {
    // terminal case 0: coord is unary, carry_in is either unary or nullary; call final_combine
    // if carry_in is (), we map that to zero
    if constexpr (nullary_coordinate<C2>)
    {
      return final_combine(0_c, detail::to_integral_like(coord));
    }
    else
    {
      return final_combine(detail::to_integral_like(carry_in), detail::to_integral_like(coord));
    }
  }
  else if constexpr (equal_rank<C1,C2>)
  {
    // recursive case 1: coord and carry_in are tuples of equal rank
    // map coordinate_inclusive_scan_with_final across coord & carry_in
    // we don't need to unzip anything because this function just returns a single result
    return tuples::zip_with(coord, carry_in, [&](auto coord_i, auto carry_i)
    {
      return coordinate_inclusive_scan_with_final(coord_i, carry_i, combine, final_combine);
    });
  }
  else
  {
    // recursive case 2: coord is multiary and carry_in is either nullary or unary
    static_assert(multiary_coordinate<C1>);
    static_assert(nullary_coordinate<C2> or unary_coordinate<C2>);

    // Some scan-like operations on coordinates need to treat the final mode of the coordinate specially
    // To accomplish this, we split the input coord into its front elements and its final element

    // We use a normal inclusive scan on the front portion. We pass the carry_in to that and receive 
    // the front portion of the result coordinate and a carry_out
    //
    // Then, we recurse on the final element of coord, and pass the carry_out to that recursion
    //
    // Finally, we return the front result appended to the last result 

    // XXX the use of unwrap/wrap here deals with a case like the following:
    // coord = ((a,b,c), (d,e))
    //
    // Dropping the last element of coord yields a degenerate
    // coord_front = ((a,b,c)) with superfluous parentheses
    // which is not currently recognized as a coordinate
    //
    // so we unwrap single-element coord_front
    // and then apply a wrapping to result_front to account for it

    auto coord_front = tuples::drop_last_and_unwrap_single(coord);
    auto coord_last  = tuples::last(coord);

    // do a normal scan on the front
    auto [result_front, carry_out] = coordinate_inclusive_scan_and_fold(coord_front, carry_in, combine);

    // recurse on the final mode
    auto result_last = coordinate_inclusive_scan_with_final(coord_last, carry_out, combine, final_combine);

    // return the tuple result_front o result_last
    return tuples::append_like<C1>(tuples::wrap_if<rank_v<C1> == 2>(result_front), result_last);
  }
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

