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
#include <utility>


namespace ubu
{
namespace detail
{


template<coordinate D, coordinate S>
constexpr congruent<S> auto compact_right_major_stride_impl(const D& prev_product, const S& shape)
{
  if constexpr (unary_coordinate<D>)
  {
    if constexpr (unary_coordinate<S>)
    {
      return to_integral_like(prev_product);
    }
    else
    {
      auto unit = tuples::make_like<S>();
      auto init = std::pair(prev_product, unit);

      auto [_,result] = tuples::fold_right(shape, init, [](auto prev, auto s)
      {
        auto [prev_product, prev_result] = prev;
        auto result = tuples::prepend_like<S>(prev_result, compact_right_major_stride_impl(prev_product, s));

        return std::pair(prev_product * shape_size(s), result);
      });

      return result;
    }
  }
  else
  {
    return tuples::zip_with(prev_product, shape, [](const auto& p, const auto& s)
    {
      return compact_left_major_stride_impl(p, s);
    });
  }
}


} // end detail


template<coordinate S>
constexpr congruent<S> auto compact_right_major_stride(const S& shape)
{
  return detail::compact_right_major_stride_impl(1_c, shape);
}

template<coordinate S>
using compact_right_major_stride_t = decltype(compact_right_major_stride(std::declval<S>()));
  

} // end ubu


#include "../../../../detail/epilogue.hpp"

