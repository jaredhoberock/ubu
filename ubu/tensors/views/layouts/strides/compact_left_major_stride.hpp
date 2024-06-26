#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/constant.hpp"
#include "../../../../miscellaneous/integrals/integral_like.hpp"
#include "../../../../miscellaneous/tuples.hpp"
#include "../../../coordinates/concepts/congruent.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/concepts/equal_rank.hpp"
#include "../../../coordinates/detail/to_integral_like.hpp"
#include "../../../coordinates/traits/rank.hpp"
#include "../../../shapes/shape_size.hpp"
#include <concepts>
#include <tuple>
#include <utility>


namespace ubu
{
namespace detail
{


template<scalar_coordinate D, scalar_coordinate S>
constexpr integral_like auto compact_left_major_stride_impl(const D& current_stride, const S&)
{
  return to_integral_like(current_stride);
}

template<nonscalar_coordinate D, nonscalar_coordinate S>
  requires equal_rank<D,S>
constexpr congruent<S> auto compact_left_major_stride_impl(const D& current_stride, const S& shape)
{
  return tuples::zip_with(current_stride, shape, [](const auto& cs, const auto& s)
  {
    return compact_left_major_stride_impl(cs, s);
  });
}

template<scalar_coordinate D, nonscalar_coordinate S>
constexpr congruent<S> auto compact_left_major_stride_impl(const D& current_stride, const S& shape)
{
  auto [_,result] = tuples::fold_left(std::pair(current_stride, std::tuple()), shape, [](auto prev, auto s)
  {
    auto [current_stride, prev_result] = prev;
    auto result = tuples::append_like<S>(prev_result, compact_left_major_stride_impl(current_stride, s));

    return std::pair{current_stride * shape_size(s), result};
  });

  return result;
}


} // end detail


template<coordinate S>
constexpr congruent<S> auto compact_left_major_stride(const S& shape)
{
  // XXX ideally, the type of the constant we use here should
  //     be the same type as coordinate_element_t<0,S>
  return detail::compact_left_major_stride_impl(1_c, shape);
}

template<coordinate S>
using compact_left_major_stride_t = decltype(compact_left_major_stride(std::declval<S>()));
  

} // end ubu::detail


#include "../../../../detail/epilogue.hpp"

