#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../event/first_cause.hpp"
#include "../asynchronous_allocator.hpp"

namespace ubu
{


template<class A>
  requires asynchronous_allocator<A>
using allocator_event_t = first_cause_result_t<A>;


} // end ubu

#include "../../../detail/epilogue.hpp"

