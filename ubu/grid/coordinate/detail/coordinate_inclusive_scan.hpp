#pragma once

#include "../../../detail/prologue.hpp"

#include "../congruent.hpp"
#include "../coordinate.hpp"
#include "../element.hpp"
#include "tuple_algorithm.hpp"
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
// When the result of the combine function's is an integer, then the result_tuple is a
// coordinate conguent with C1.
//
// The combination function has control over the value of the carry after each combination
// combine(input, carry_in) must return the pair (result, carry_out)
//
// The types of combine's input and carry_in parameters must be integers.
// The type of combine's carry_out result must be an integer.

template<scalar_coordinate C1, scalar_coordinate C2, class F>
constexpr pair_like auto coordinate_inclusive_scan(const C1& coord, const C2& carry_in, const F& combine)
{
  // the use of element<0>(...) ensures that we pass integers to combine
  return combine(element<0>(coord), element<0>(carry_in));
}

template<nonscalar_coordinate C1, scalar_coordinate C2, class F>
constexpr pair_like auto coordinate_inclusive_scan(const C1& coord, const C2& carry_in, const F& combine)
{
  return tuple_inclusive_scan_with_carry(coord, carry_in, [&](auto coord_i, auto carry_i)
  {
    return coordinate_inclusive_scan(coord_i, carry_i, combine);
  });
}

// the returned carry is congruent with C2
template<nonscalar_coordinate C1, nonscalar_coordinate C2, class F>
  requires weakly_congruent<C2,C1>
constexpr pair_like auto coordinate_inclusive_scan(const C1& coord, const C2& carry_in, const F& combine)
{
  return tuple_unzip(tuple_zip_with(coord, carry_in), [&](const auto& coord_i, const auto& carry_i)
  {
    return coordinate_inclusive_scan(coord_i, carry_i, combine);
  });
}


// coordinate_inclusive_scan_with_final is like coordinate_inclusive_scan except that it treats the final combination operation in a special way
// it yields a single result which is a tuple the same size as the input coordinate's rank
// when the combine function yields an integer result, then the result of coordinate_inclusive_scan_with_final is a coordinate congruent with
// the input coordinate.

template<scalar_coordinate C1, scalar_coordinate C2, class F1, class F2>
constexpr auto coordinate_inclusive_scan_with_final(const C1& coord, const C2& carry_in, const F1&, const F2& final_combine)
{
  // when both the coord and carry_in arguments are scalars, then this is the final combination operation
  // just return its result
  return final_combine(coord, carry_in);
}

// XXX if the combine function is allowed to return anything, then the result of this function may not be a coordinate
template<nonscalar_coordinate C1, scalar_coordinate C2, class F1, class F2>
constexpr tuple_like_of_size<rank_v<C1>> auto coordinate_inclusive_scan_with_final(const C1& coord, const C2& carry_in, const F1& combine, const F2& final_combine)
{
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
  auto coord_front = tuple_unwrap_single(tuple_drop_last(coord));
  auto coord_last  = tuple_last(coord);

  // do a normal scan on the front
  auto [result_front, carry_out] = coordinate_inclusive_scan(coord_front, carry_in, combine);

  // recurse on the final mode
  auto result_last = coordinate_inclusive_scan_with_final(coord_last, carry_out, combine, final_combine);

  // return the tuple result_front o result_last
  return tuple_append_similar_to<C1>(tuple_wrap_if<rank_v<C1> == 2>(result_front), result_last);
}

template<nonscalar_coordinate C1, nonscalar_coordinate C2, class F1, class F2>
  requires (not congruent<C1,C2> and weakly_congruent<C2,C1>)
constexpr tuple_like_of_size<rank_v<C1>> auto coordinate_inclusive_scan_with_final(const C1& coord, const C2& carry_in, const F1& combine, const F2& final_combine)
{
  return tuple_zip_with(coord, carry_in, [&](const auto& coord_i, const auto& carry_in_i)
  {
    return coordinate_inclusive_scan_with_final(coord_i, carry_in_i, combine, final_combine);
  });
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

