#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/coordinate.hpp"
#include "strided_layout.hpp"
#include "stride/compact_column_major_stride.hpp"

namespace ubu
{


template<coordinate S>
struct column_major : public strided_layout<S,S>
{
  constexpr column_major(S shape)
    : strided_layout<S,S>{shape, compact_column_major_stride(shape)}
  {}

  column_major(const column_major&) = default;
};


} // end ubu

#include "../../detail/epilogue.hpp"

