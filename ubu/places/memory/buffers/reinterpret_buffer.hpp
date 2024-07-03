#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/constant.hpp"
#include "../../../miscellaneous/integrals/integral_like.hpp"
#include "../../../miscellaneous/integrals/size.hpp"
#include "../../../tensors/vectors/fancy_span.hpp"
#include "../../../tensors/vectors/span_like.hpp"
#include "../data.hpp"
#include "../pointers/reinterpret_pointer.hpp"
#include "buffer_like.hpp"

namespace ubu
{

template<class T, buffer_like B, integral_like N>
constexpr span_like auto reinterpret_buffer(B buffer, N num_objects)
{
  return fancy_span(reinterpret_pointer<T>(data(buffer)), num_objects);
}

template<class T, buffer_like B>
constexpr span_like auto reinterpret_buffer(B buffer)
{
  auto num_objects = size(buffer) / constant<sizeof(T)>();

  return reinterpret_buffer<T>(buffer, num_objects);
}

} // end ubu

#include "../../../detail/epilogue.hpp"

