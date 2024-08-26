#pragma once

#include "../../../../detail/prologue.hpp"

#include "../concepts/allocator.hpp"
#include <type_traits>

namespace ubu
{


template<allocator A>
using allocator_value_t = typename std::remove_cvref_t<A>::value_type;


} // end ubu

#include "../../../../detail/epilogue.hpp"

