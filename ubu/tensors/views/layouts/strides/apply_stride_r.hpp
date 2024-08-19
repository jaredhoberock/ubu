#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../utilities/tuples.hpp"
#include "../../../coordinates/concepts/congruent.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/concepts/weakly_congruent.hpp"
#include "../../../coordinates/coordinate_sum.hpp"
#include "../../../coordinates/detail/to_integral_like.hpp"
#include "../../../coordinates/traits/zeros.hpp"
#include "apply_stride.hpp"


namespace ubu
{


// this variant of apply_stride allows control over the width of the result type, R
// R must be congruent to the result returned by the other variant, apply_stride()
template<coordinate R, coordinate D, weakly_congruent<D> C>
  requires (congruent<R, apply_stride_result_t<D,C>> and not std::is_reference_v<R>)
constexpr R apply_stride_r(const D& stride, const C& coord)
{
  return coordinate_product_r<R>(coord, stride);
}


} // end ubu

#include "../../../../detail/epilogue.hpp"

