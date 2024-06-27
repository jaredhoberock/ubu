#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "compact_left_major.hpp"

namespace ubu
{


template<coordinate_of_rank<2> S>
struct column_major : public compact_left_major<S>
{
  constexpr column_major(S shape)
    : compact_left_major<S>(shape)
  {}

  column_major(const column_major&) = default;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

