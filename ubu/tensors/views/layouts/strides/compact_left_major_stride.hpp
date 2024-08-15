#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../utilities/constant.hpp"
#include "../../../../utilities/integrals/integral_like.hpp"
#include "../../../../utilities/tuples.hpp"
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


template<coordinate D, coordinate S>
constexpr congruent<S> auto compact_left_major_stride_impl(const D& current_stride, const S& shape)
{
  if constexpr (unary_coordinate<D>)
  {
    if constexpr (unary_coordinate<S>)
    {
      return to_integral_like(current_stride);
    }
    else
    {
      auto unit = tuples::make_like<S>();
      auto init = std::pair(current_stride, unit);

      auto [_,result] = tuples::fold_left(shape, init, [](auto prev, auto s)
      {
        auto [current_stride, prev_result] = prev;
        auto result = tuples::append_like<S>(prev_result, compact_left_major_stride_impl(current_stride, s));

        return std::pair(current_stride * shape_size(s), result);
      });

      return result;
    }
  }
  else
  {
    return tuples::zip_with(current_stride, shape, [](const auto& cs, const auto& s)
    {
      return compact_left_major_stride_impl(cs, s);
    });
  }
}


} // end detail


template<coordinate S>
constexpr congruent<S> auto compact_left_major_stride(const S& shape)
{
  return detail::compact_left_major_stride_impl(1_c, shape);
}

template<coordinate S>
using compact_left_major_stride_t = decltype(compact_left_major_stride(std::declval<S>()));
  

} // end ubu::detail


#include "../../../../detail/epilogue.hpp"

