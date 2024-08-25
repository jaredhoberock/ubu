#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/vectors/span_like.hpp"
#include "../../data.hpp"
#include "../concepts/allocator.hpp"
#include "allocator_value.hpp"
#include "allocator_view.hpp"

namespace ubu
{

// XXX we should eliminate this trait because allocators that return a span are a special case
template<allocator A, class T = allocator_value_t<A>>
  requires span_like<allocator_view_t<A,T>>
using allocator_pointer_t = data_t<allocator_view_t<A,T>>;

} // end ubu

#include "../../../../detail/epilogue.hpp"

