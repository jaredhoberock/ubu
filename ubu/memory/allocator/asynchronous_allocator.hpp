#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/first_cause.hpp"
#include "../../causality/happening.hpp"
#include "allocate_after.hpp"
#include "allocator.hpp"
#include "deallocate_after.hpp"
#include "traits/allocator_pointer_t.hpp"
#include "traits/allocator_value_t.hpp"
#include <memory>
#include <type_traits>

namespace ubu
{

template<class A, class T>
concept asynchronous_allocator_of =
  allocator_of<A,T>

  and requires(A a)
  {
    {first_cause(a)} -> happening;
  }

  and requires(A a, const first_cause_result_t<A>& before, allocator_pointer_t<A,T> ptr, std::size_t n)
  {
    // XXX this needs to check that the result is a pair<happening,pointer>
    ubu::allocate_after<T>(a, before, n);
  
    {ubu::deallocate_after(a, before, ptr, n)} -> happening;
  }
;

template<class A>
concept asynchronous_allocator = allocator<A> and asynchronous_allocator_of<A,int>;

} // end ubu

#include "../../detail/epilogue.hpp"

