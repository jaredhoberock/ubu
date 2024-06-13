#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../causality/initial_happening.hpp"
#include "../concepts/asynchronous_allocator.hpp"

namespace ubu
{


template<asynchronous_allocator A>
using allocator_happening_t = initial_happening_result_t<A>;


} // end ubu

#include "../../../../detail/epilogue.hpp"

