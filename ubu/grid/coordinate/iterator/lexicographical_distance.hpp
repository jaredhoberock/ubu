#pragma once

#include "../../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "../congruent.hpp"
#include "../coordinate_difference.hpp"
#include "../../layout/stride/apply_stride.hpp"
#include "../../layout/stride/compact_row_major_stride.hpp"
#include <concepts>

namespace ubu
{


template<coordinate C1, congruent<C1> C2, congruent<C1> S>
constexpr std::integral auto lexicographical_distance(const C1& from, const C2& to, const S& shape)
{
  auto delta = coordinate_difference(to, from);
  return apply_stride(delta, compact_row_major_stride(shape));
}


} // end ubu


#include "../../../detail/epilogue.hpp"
