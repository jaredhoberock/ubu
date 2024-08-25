#pragma once

#include "../../../../detail/prologue.hpp"

#include "../concepts/allocator.hpp"
#include <type_traits>

namespace ubu
{

template<allocator A>
using allocator_shape_t = typename detail::allocator_shape<std::remove_cvref_t<A>>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

