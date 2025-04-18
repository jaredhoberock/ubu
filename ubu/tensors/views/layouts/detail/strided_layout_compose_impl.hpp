#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../utilities/tuples.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "compatible_shape.hpp"
#include "compose_strides.hpp"
#include "divide_shape.hpp"
#include <algorithm>


namespace ubu::detail
{

// strided_layout_compose_impl is the implementation of strided_layout::compose
// It returns the pair (composition_shape, composition_stride)

// case 1: scalar RShape
template<coordinate LStride, weakly_congruent<LStride> LShape,
         scalar_coordinate RShape, coordinate RStride>
constexpr auto strided_layout_compose_impl(const LShape& lhs_shape, const LStride& lhs_stride,
                                           const RShape& rhs_shape, const RStride& rhs_stride)
{
  auto [result_shape_1, divisors] = divide_shape(lhs_shape, rhs_stride);
  auto result_shape = compatible_shape(result_shape_1, rhs_shape);
  auto result_stride = compose_strides(divisors, lhs_stride);

  return std::pair(result_shape, result_stride);
}

// case 2: nonscalar RShape
template<coordinate LShape, equal_rank<LShape> LStride,
         nonscalar_coordinate RShape, equal_rank<RShape> RStride>
constexpr auto strided_layout_compose_impl(const LShape& lhs_shape, const LStride& lhs_stride,
                                           const RShape& rhs_shape, const RStride& rhs_stride)
{
  auto tuple_of_pairs = tuples::zip_with(rhs_shape, rhs_stride, [&](const auto& s, const auto& d)
  {
    return strided_layout_compose_impl(lhs_shape, lhs_stride, s, d);
  });

  return tuples::unzip(tuple_of_pairs);
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

