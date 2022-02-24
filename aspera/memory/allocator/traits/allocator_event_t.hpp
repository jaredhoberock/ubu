#pragma once

#include "../../../detail/prologue.hpp"

#include "../asynchronous_deallocator.hpp"

ASPERA_NAMESPACE_OPEN_BRACE


template<class A>
  requires asynchronous_deallocator<A>
using allocator_event_t = typename std::decay_t<A>::event_type;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"
