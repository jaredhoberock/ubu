#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/event.hpp"
#include "deallocate_after.hpp"
#include <memory>
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE

template<class A>
concept asynchronous_deallocator =
  // allocator<A> and

  requires{typename std::remove_cvref_t<A>::event_type; } and

  event<typename std::remove_cvref_t<A>::event_type> and

  requires(A a, const typename std::remove_cvref_t<A>::event_type& e, typename std::allocator_traits<std::remove_cvref_t<A>>::pointer ptr, std::size_t n)
  {
    {ASPERA_NAMESPACE::deallocate_after(a, e, ptr, n)} -> event;
  }
;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"
