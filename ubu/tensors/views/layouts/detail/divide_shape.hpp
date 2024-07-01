#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/integrals/ceil_div.hpp"
#include "../../../../miscellaneous/tuples.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/detail/coordinate_inclusive_scan.hpp"
#include <utility>


namespace ubu::detail
{

// divide_shape returns a pair of coordinates (quotient, divisor), where:
// 1. quotient is a coordinate that is the result of dividing numerator by denominator, and
// 2. divisor is a tuple of integers, where each element contains the divisor used to produce
//    the corresponding element of quotient
template<coordinate N, scalar_coordinate D>
constexpr tuples::pair_like auto divide_shape(const N& numerator, const D& denominator)
{
  // combine's primary result is the quotient of n and d, along with the divisor used in the quotient
  // the carry is the reciprocal of this quotient
  auto combine = [](auto n, auto d)
  {
    auto quotient = ceil_div(n,d);
    auto divisor = n < d ? n : d;
    std::pair result(quotient, divisor);

    // the carry is the reciprocal of the ceil_div we just did
    auto carry = ceil_div(d,n);

    return std::pair(result, carry);
  };

  // the final combination just returns the arguments
  // it doesn't do any division on the final mode
  auto final_combine = [](auto n, auto d)
  {
    return std::pair(n,d);
  };

  // this scan will return a nested tuple whose innermost elements are pairs (quotient_i, divisor_i)
  auto almost_result = coordinate_inclusive_scan_with_final(numerator, denominator, combine, final_combine);

  // unzip the innermost pairs to yield (quotient, divisor)
  return tuples::unzip_innermost_pairs(almost_result);
}


} // end ubu::detail


#include "../../../../detail/epilogue.hpp"

