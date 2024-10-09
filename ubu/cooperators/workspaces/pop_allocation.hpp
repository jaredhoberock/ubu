#pragma once

#include "../../detail/prologue.hpp"
#include "../../places/memory/views/get_buffer.hpp"
#include "detail/advance_data.hpp"

namespace ubu
{

// XXX ideally, we would make this work for any memory_view
template<workspace W>
  requires detail::advanceable_span_like<buffer_t<W>,int>
constexpr void pop_allocation(W& ws, int n)
{
  detail::advance_data(get_buffer(ws), -n);
}

} // end ubu

#include "../../detail/epilogue.hpp"

