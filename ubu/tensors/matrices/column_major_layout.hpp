#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/concepts/coordinate.hpp"
#include "../views/layouts/compact_left_major_layout.hpp"

namespace ubu
{


template<coordinate_of_rank<2> S>
struct column_major_layout : public compact_left_major_layout<S>
{
  constexpr column_major_layout(S shape)
    : compact_left_major_layout<S>(shape)
  {}

  column_major_layout(const column_major_layout&) = default;
};


} // end ubu

#include "../../detail/epilogue.hpp"

