#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../../miscellaneous/tuples.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/point.hpp"
#include "../views/layouts/compact_right_major_layout.hpp"
#include <utility>

namespace ubu
{


template<coordinate_of_rank<2> S>
struct row_major_layout : public compact_right_major_layout<S>
{
  constexpr row_major_layout(S shape)
    : compact_right_major_layout<S>(shape)
  {}

  constexpr row_major_layout(tuples::first_t<S> num_rows, tuples::second_t<S> num_columns)
    : row_major_layout(S(num_rows, num_columns))
  {}

  row_major_layout(const row_major_layout&) = default;
};

template<integral_like I>
row_major_layout(I,I) -> row_major_layout<point<I,2>>;

template<integral_like R, integral_like C>
row_major_layout(R,C) -> row_major_layout<std::pair<R,C>>;


} // end ubu

#include "../../detail/epilogue.hpp"

