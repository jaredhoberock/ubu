#pragma once

#include "../../../../detail/prologue.hpp"

#include "../concepts/allocator.hpp"
#include <type_traits>

namespace ubu
{

template<allocator A>
using allocator_shape_t = detail::maybe_allocator_shape_t<A>;

} // end ubu

#include "../../../../detail/epilogue.hpp"

