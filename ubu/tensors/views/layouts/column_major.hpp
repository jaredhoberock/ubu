#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "strided_layout.hpp"
#include "strides/compact_column_major_stride.hpp"

namespace ubu
{


template<coordinate S>
struct column_major : public strided_layout<S,compact_column_major_stride_t<S>>
{
  constexpr column_major(S shape)
    : strided_layout<S,compact_column_major_stride_t<S>>{shape, compact_column_major_stride(shape)}
  {}

  column_major(const column_major&) = default;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

