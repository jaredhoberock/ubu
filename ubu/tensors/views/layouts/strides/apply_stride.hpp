#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../coordinates/concepts/weakly_congruent.hpp"
#include "../../../coordinates/coordinate_weak_product.hpp"
#include <utility>


namespace ubu
{


template<coordinate D, weakly_congruent<D> C>
constexpr coordinate auto apply_stride(const D& stride, const C& coord)
{
  return coordinate_weak_product(coord,stride);
}


template<coordinate D, weakly_congruent<D> C>
using apply_stride_result_t = decltype(apply_stride(std::declval<D>(), std::declval<C>()));


} // end ubu

#include "../../../../detail/epilogue.hpp"

