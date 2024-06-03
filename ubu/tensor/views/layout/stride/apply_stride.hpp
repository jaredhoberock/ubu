#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../coordinate/concepts/coordinate.hpp"
#include "../../../coordinate/coordinate_sum.hpp"
#include "../../../coordinate/detail/as_integral_like.hpp"
#include "../../../coordinate/detail/tuple_algorithm.hpp"
#include <concepts>
#include <utility>


namespace ubu
{
namespace detail
{


template<scalar_coordinate D, scalar_coordinate C>
constexpr integral_like auto apply_stride_impl(const D& stride, const C& coord)
{
  return as_integral_like(stride) * as_integral_like(coord);
}


template<nonscalar_coordinate D, nonscalar_coordinate C>
  requires weakly_congruent<C,D>
constexpr coordinate auto apply_stride_impl(const D& stride, const C& coord);


template<nonscalar_coordinate D, scalar_coordinate C>
constexpr congruent<D> auto apply_stride_impl(const D& stride, const C& coord)
{
  return detail::tuple_zip_with(stride, [&](const auto& s)
  {
    return apply_stride_impl(s, coord);
  });
}


template<nonscalar_coordinate D, nonscalar_coordinate C>
  requires weakly_congruent<C,D>
constexpr coordinate auto apply_stride_impl(const D& stride, const C& coord)
{
  auto star = [](const auto& s, const auto& c)
  {
    return apply_stride_impl(s,c);
  };

  auto plus = [](const auto& c1, const auto& c2)
  {
    return coordinate_sum(c1,c2);
  };

  return detail::tuple_inner_product(stride, coord, star, plus);
}


} // end detail


template<coordinate D, weakly_congruent<D> C>
constexpr coordinate auto apply_stride(const D& stride, const C& coord)
{
  return detail::apply_stride_impl(stride, coord);
}


template<coordinate D, weakly_congruent<D> C>
using apply_stride_t = decltype(apply_stride(std::declval<D>(), std::declval<C>()));


} // end ubu

#include "../../../../detail/epilogue.hpp"

