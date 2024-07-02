#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../../miscellaneous/tuples.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/point.hpp"
#include "../views/layouts/compact_left_major_layout.hpp"
#include <utility>

namespace ubu
{


template<coordinate_of_rank<2> S>
struct column_major_layout : public compact_left_major_layout<S>
{
  constexpr column_major_layout(S shape)
    : compact_left_major_layout<S>(shape)
  {}

  constexpr column_major_layout(tuples::first_t<S> num_rows, tuples::second_t<S> num_columns)
    : column_major_layout(S(num_rows, num_columns))
  {}

  column_major_layout(const column_major_layout&) = default;
};

template<integral_like I>
column_major_layout(I,I) -> column_major_layout<point<I,2>>;

template<integral_like R, integral_like C>
column_major_layout(R,C) -> column_major_layout<std::pair<R,C>>;


} // end ubu

#include "../../detail/epilogue.hpp"

