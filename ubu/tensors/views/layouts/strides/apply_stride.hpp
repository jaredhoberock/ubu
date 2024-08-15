#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../utilities/tuples.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/coordinate_sum.hpp"
#include "../../../coordinates/detail/to_integral_like.hpp"
#include <concepts>
#include <utility>


namespace ubu
{


template<coordinate D, weakly_congruent<D> C>
  requires nonnullary_coordinate<C>
constexpr coordinate auto apply_stride(const D& stride, const C& coord)
{
  if constexpr (unary_coordinate<C>)
  {
    if constexpr (unary_coordinate<D>)
    {
      // both stride & coord are integral, just multiply
      return detail::to_integral_like(stride) * detail::to_integral_like(coord);
    }
    else
    {
      // stride is a tuple, map apply_stride across it
      return tuples::zip_with(stride, [&](const auto& s_i)
      {
        return apply_stride(s_i, coord);
      });
    }
  }
  else
  {
    // stride & coord are non-empty tuples of the same rank, inner_product
    static_assert(tuples::tuple_like_of_size_at_least<D,1> and tuples::same_size<D,C>);

    auto star = [](const auto& s_i, const auto& c_i)
    {
      return apply_stride(s_i,c_i);
    };

    auto plus = [](const auto& c1, const auto& c2)
    {
      return coordinate_sum(c1,c2);
    };

    auto init = star(get<0>(stride), get<0>(coord));

    return tuples::inner_product(tuples::drop_first(stride), tuples::drop_first(coord), init, star, plus);
  }
}


template<coordinate D, weakly_congruent<D> C>
  requires nonnullary_coordinate<C>
using apply_stride_result_t = decltype(apply_stride(std::declval<D>(), std::declval<C>()));


} // end ubu

#include "../../../../detail/epilogue.hpp"

