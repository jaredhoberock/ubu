#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "compact_right_major.hpp"

namespace ubu
{


template<coordinate_of_rank<2> S>
struct row_major : public compact_right_major<S>
{
  constexpr row_major(S shape)
    : compact_right_major<S>(shape)
  {}

  row_major(const row_major&) = default;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

