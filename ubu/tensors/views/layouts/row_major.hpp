#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "strided_layout.hpp"
#include "strides/compact_right_major_stride.hpp"

namespace ubu
{


template<coordinate S>
struct row_major : public strided_layout<S,compact_right_major_stride_t<S>>
{
  constexpr row_major(S shape)
    : strided_layout<S,compact_right_major_stride_t<S>>{shape, compact_right_major_stride(shape)}
  {}

  row_major(const row_major&) = default;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

