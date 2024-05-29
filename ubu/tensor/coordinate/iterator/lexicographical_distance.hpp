#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/integral/integral_like.hpp"
#include "../../layout/stride/apply_stride.hpp"
#include "../../layout/stride/compact_row_major_stride.hpp"
#include "../concepts/coordinate.hpp"
#include "../concepts/congruent.hpp"
#include "../coordinate_difference.hpp"

namespace ubu
{


template<coordinate C1, congruent<C1> C2, congruent<C1> S>
constexpr integral_like auto lexicographical_distance(const C1& from, const C2& to, const S& shape)
{
  auto delta = coordinate_difference(to, from);
  return apply_stride(compact_row_major_stride(shape), delta);
}


} // end ubu


#include "../../../detail/epilogue.hpp"

