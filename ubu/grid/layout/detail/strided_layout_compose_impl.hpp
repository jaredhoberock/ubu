#pragma once

#include "../../../detail/prologue.hpp"

#include "ceil_div.hpp"
#include <algorithm>
#include <ubu/grid/coordinate/coordinate.hpp>
#include <ubu/grid/coordinate/detail/tuple_algorithm.hpp>
#include <ubu/grid/layout/detail/compatible_shape.hpp>
#include <ubu/grid/layout/detail/compose_strides.hpp>
#include <ubu/grid/layout/detail/divide_shape.hpp>


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
template<coordinate LShape, same_rank<LShape> LStride,
         nonscalar_coordinate RShape, same_rank<RShape> RStride>
constexpr auto strided_layout_compose_impl(const LShape& lhs_shape, const LStride& lhs_stride,
                                           const RShape& rhs_shape, const RStride& rhs_stride)
{
  auto tuple_of_pairs = tuple_zip_with(rhs_shape, rhs_stride, [&](const auto& s, const auto& d)
  {
    return strided_layout_compose_impl(lhs_shape, lhs_stride, s, d);
  });

  return tuple_unzip(tuple_of_pairs);
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

