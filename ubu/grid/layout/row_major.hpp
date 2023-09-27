#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/coordinate.hpp"
#include "strided_layout.hpp"
#include "stride/compact_row_major_stride.hpp"

namespace ubu
{


template<coordinate S>
struct row_major : public strided_layout<S,S>
{
  constexpr row_major(S shape)
    : strided_layout<S,S>{shape, compact_row_major_stride(shape)}
  {}

  row_major(const row_major&) = default;
};


} // end ubu

#include "../../detail/epilogue.hpp"

