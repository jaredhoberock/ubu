#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/concepts/coordinate.hpp"
#include "strided_layout.hpp"
#include "strides/compact_left_major_stride.hpp"

namespace ubu
{


template<coordinate S>
struct compact_left_major_layout : public strided_layout<S,compact_left_major_stride_t<S>>
{
  constexpr compact_left_major_layout(S shape)
    : strided_layout<S,compact_left_major_stride_t<S>>{shape, compact_left_major_stride(shape)}
  {}

  compact_left_major_layout(const compact_left_major_layout&) = default;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

