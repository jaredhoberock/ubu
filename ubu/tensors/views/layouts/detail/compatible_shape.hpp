#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../utilities/integrals/ceil_div.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/concepts/weakly_congruent.hpp"
#include "../../../coordinates/detail/coordinate_inclusive_scan.hpp"
#include <utility>


namespace ubu::detail
{


// compatible_shape converts a shape to be compatible with another weakly_congruent shape
// postcondition: shape_size(result) == shape_size(n)
template<ubu::coordinate S1, ubu::scalar_coordinate S2>
  requires weakly_congruent<S2,S1>
constexpr S1 compatible_shape(const S1& shape, const S2& n)
{
  // the combine operation returns the smaller of s and n
  // the carry of this operation is the "unused portion" of n
  auto combine = [](auto s, auto n)
  {
    using namespace std;
    auto kept = s < n ? s : n;
    return pair(kept, ceil_div(n, kept));
  };

  // the final combine operation just sets the result to n,
  // because when n is an integer, the only shape compatible with n is n
  auto final_combine = [](auto s, auto n)
  {
    return n;
  };

  return coordinate_inclusive_scan_with_final(shape, n, combine, final_combine);
}


} // end ubu::detail


#include "../../../../detail/epilogue.hpp"

