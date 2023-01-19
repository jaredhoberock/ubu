#pragma once

#include "../../../detail/prologue.hpp"

#include "ceil_div.hpp"
#include <algorithm>
#include <ubu/grid/coordinate/coordinate.hpp>
#include <ubu/grid/coordinate/detail/tuple_algorithm.hpp>
#include <ubu/grid/layout/detail/compose_strides.hpp>
#include <ubu/grid/layout/detail/divide_coordinate.hpp>
#include <ubu/grid/layout/detail/divide_shape.hpp>


namespace ubu::detail
{


// returns the pair (modulo, remainder)
template<coordinate C1, coordinate C2>
  requires weakly_congruent<C2,C1>
constexpr auto modulo(const C1& dividend, const C2& divisor)
{
  using namespace std;

  return divide_coordinate(dividend, divisor, [](int dividend, int divisor)
  {
    return pair(min(dividend,divisor), ceil_div(divisor, dividend));
  });
}


// scalar everything
// returns the pair (composition_shape, composition_stride)
template<scalar_coordinate LShape, scalar_coordinate LStride,
         scalar_coordinate RShape, scalar_coordinate RStride>
constexpr auto strided_layout_compose_impl(const LShape&, const LStride& lhs_stride,
                                           const RShape& rhs_shape, const RStride& rhs_stride)
{
  auto result_stride = rhs_stride * lhs_stride;
  return std::pair(rhs_shape, result_stride);
}


// scalar RShape
template<nonscalar_coordinate LStride, weakly_congruent<LStride> LShape,
         scalar_coordinate RShape, coordinate RStride>
constexpr auto strided_layout_compose_impl(const LShape& lhs_shape, const LStride& lhs_stride,
                                           const RShape& rhs_shape, const RStride& rhs_stride)
{
  // XXX how do we eliminate the unused residual stuff?
  auto [result_shape_1, denominator, _] = divide_shape(lhs_shape, rhs_stride);
  auto [result_shape, __] = modulo(result_shape_1, rhs_shape);
  auto result_stride = compose_strides(denominator, lhs_stride);

  return std::pair(result_shape, result_stride);
}


// nonscalar B shape
// returns the pair (composition_shape, composition_stride)
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

