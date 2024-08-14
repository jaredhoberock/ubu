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
  if constexpr (scalar_coordinate<C>)
  {
    if constexpr (scalar_coordinate<D>)
    {
      // both stride & coord are integral, just cast the result of apply_stride
      return static_cast<R>(apply_stride(stride, coord));
    }
    else
    {
      // stride is a tuple, map apply_stride_r across it
      return tuples::zip_with(stride, zeros_v<R>, [&](const auto& s_i, auto r_i)
      {
        return apply_stride_r<decltype(r_i)>(s_i, coord);
      });
    }
  }
  else
  {
    // stride & coord are tuples of the same rank, inner_product
    static_assert(tuples::tuple_like<D> and tuples::tuple_like<C>);
    static_assert(equal_rank<D,C>);

    auto star = [](const auto& s_i, const auto& c_i)
    {
      return apply_stride_r<R>(s_i,c_i);
    };

    auto plus = [](const auto& c1, const auto& c2)
    {
      return coordinate_sum(c1,c2);
    };

    return tuples::inner_product(stride, coord, star, plus);
  }
}


} // end ubu

#include "../../../../detail/epilogue.hpp"

