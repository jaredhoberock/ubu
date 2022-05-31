#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../event/make_independent_event.hpp"
#include "../asynchronous_allocator.hpp"

UBU_NAMESPACE_OPEN_BRACE


template<class A>
  requires asynchronous_allocator<A>
using allocator_event_t = make_independent_event_result_t<A>;


UBU_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

