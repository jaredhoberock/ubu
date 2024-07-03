#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/constant.hpp"
#include "../../../miscellaneous/integrals/integral_like.hpp"
#include "../../../miscellaneous/integrals/size.hpp"
#include "../../../tensors/vectors/fancy_span.hpp"
#include "../../../tensors/vectors/span_like.hpp"
#include "../data.hpp"
#include "buffer_like.hpp"
#include <cstddef>

namespace ubu
{

template<class T, buffer_like B, integral_like N>
constexpr fancy_span<T*,N> reinterpret_buffer(B buffer, N num_objects)
{
  // XXX data returns pointer_like, so we probably need something more like to_raw_pointer
  return {reinterpret_cast<T*>(data(buffer)), num_objects};
}

template<class T, buffer_like B>
constexpr span_like auto reinterpret_buffer(B buffer)
{
  auto num_objects = size(buffer) / constant<sizeof(T)>();

  return reinterpret_buffer<T>(buffer, num_objects);
}

} // end ubu

#include "../../../detail/epilogue.hpp"

