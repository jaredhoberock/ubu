#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../event/make_independent_event.hpp"
#include "../asynchronous_allocator.hpp"

namespace ubu
{


template<class A>
  requires asynchronous_allocator<A>
using allocator_event_t = make_independent_event_result_t<A>;


} // end ubu

#include "../../../detail/epilogue.hpp"

