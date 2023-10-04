#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/first_cause.hpp"
#include "../concepts/asynchronous_allocator.hpp"

namespace ubu
{


template<asynchronous_allocator A>
using allocator_happening_t = first_cause_result_t<A>;


} // end ubu

#include "../../../detail/epilogue.hpp"

