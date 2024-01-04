#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/concepts/coordinate.hpp"
#include "../../coordinate/coordinate_sum.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include "../../coordinate/element.hpp"
#include "stride_for.hpp"
#include <concepts>


namespace ubu
{
namespace detail
{


template<scalar_coordinate C, scalar_coordinate D>
constexpr std::integral auto apply_stride_impl(const C& coord, const D& stride)
{
  return element<0>(stride) * element<0>(coord);
}


template<nonscalar_coordinate C, nonscalar_coordinate D>
  requires weakly_congruent<C,D>
constexpr coordinate auto apply_stride_impl(const C& coord, const D& stride);


template<scalar_coordinate C, nonscalar_coordinate D>
constexpr congruent<D> auto apply_stride_impl(const C& coord, const D& stride)
{
  return detail::tuple_zip_with(stride, [&](const auto& s)
  {
    return apply_stride_impl(coord, s);
  });
}


template<nonscalar_coordinate C, nonscalar_coordinate D>
  requires weakly_congruent<C,D>
constexpr coordinate auto apply_stride_impl(const C& coord, const D& stride)
{
  auto star = [](const auto& c, const auto& s)
  {
    return apply_stride_impl(c,s);
  };

  auto plus = [](const auto& c1, const auto& c2)
  {
    return coordinate_sum(c1,c2);
  };

  return detail::tuple_inner_product(coord, stride, star, plus);
}


} // end detail


template<coordinate C, stride_for<C> D>
constexpr coordinate auto apply_stride(const C& coord, const D& stride)
{
  return detail::apply_stride_impl(coord, stride);
}


} // end ubu

#include "../../../detail/epilogue.hpp"

