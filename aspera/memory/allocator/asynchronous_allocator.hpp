#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "../../event/make_independent_event.hpp"
#include "allocate_after.hpp"
#include "allocator.hpp"
#include "deallocate_after.hpp"
#include "traits/allocator_pointer_t.hpp"
#include "traits/allocator_value_t.hpp"
#include <memory>
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE

template<class A>
concept asynchronous_allocator =
  allocator<A>

  and requires(A a)
  {
    {make_independent_event(a)} -> event;
  }

  and requires(A a, const make_independent_event_result_t<A>& e, allocator_pointer_t<A> ptr, std::size_t n)
  {
    // XXX this needs to check that the result is a pair<event,pointer>
    ASPERA_NAMESPACE::allocate_after<allocator_value_t<A>>(a, e, n);
  
    {ASPERA_NAMESPACE::deallocate_after(a, e, ptr, n)} -> event;
  }
;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

