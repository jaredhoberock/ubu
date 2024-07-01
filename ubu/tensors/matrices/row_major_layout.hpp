#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/concepts/coordinate.hpp"
#include "../views/layouts/compact_right_major_layout.hpp"

namespace ubu
{


template<coordinate_of_rank<2> S>
struct row_major_layout : public compact_right_major_layout<S>
{
  constexpr row_major_layout(S shape)
    : compact_right_major_layout<S>(shape)
  {}

  row_major_layout(const row_major_layout&) = default;
};


} // end ubu

#include "../../detail/epilogue.hpp"

