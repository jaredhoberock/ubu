#pragma once

#include "../../detail/prologue.hpp"
#include "../../places/memory/data.hpp"
#include "../../places/memory/views/get_buffer.hpp"
#include "detail/advance_data.hpp"

namespace ubu
{

// XXX ideally, we would make this work for any memory_view
//     consider returning a contiguous span of bytes instead of a pointer
template<workspace W>
  requires detail::advanceable_span_like<buffer_t<W>,int>
constexpr pointer_like auto push_allocation(W& ws, int n)
{
  buffer_like auto& buffer = get_buffer(ws);
  pointer_like auto result = data(buffer);
  detail::advance_data(buffer, n);
  return result;
}

} // end ubu

#include "../../detail/epilogue.hpp"

