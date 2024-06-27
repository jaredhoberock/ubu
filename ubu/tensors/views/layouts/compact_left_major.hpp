#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "strided_layout.hpp"
#include "strides/compact_left_major_stride.hpp"

namespace ubu
{


template<coordinate S>
struct compact_left_major : public strided_layout<S,compact_left_major_stride_t<S>>
{
  constexpr compact_left_major(S shape)
    : strided_layout<S,compact_left_major_stride_t<S>>{shape, compact_left_major_stride(shape)}
  {}

  compact_left_major(const compact_left_major&) = default;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

