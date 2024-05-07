#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include "../../../causality/initial_happening.hpp"
#include "../../../tensor/fancy_span.hpp"
#include "../allocate_after.hpp"
#include "../deallocate_after.hpp"
#include "../traits/allocator_pointer_t.hpp"
#include "../traits/allocator_value_t.hpp"
#include "allocator.hpp"
#include <cstddef>
#include <memory>
#include <type_traits>

namespace ubu
{

// XXX a new, tensor-based asynchronous_allocator_of concept should look something like this:
//
// template<class A, class T>
// concept asynchronous_allocator_of =
//   requires(A a)
//   {
//     {initial_happening(a)} -> happening;
//   }
//
//   and requires(A a, const initial_happening_result_t<A>& before, T tensor, tensor_shape_t<A> shape)
//   {
//     allocate_after<tensor_element_t<A>>(a, before, shape);
//     requires std::same_as<tensor_element_t<A>, allocate_after_result_t<...>);
//     deallocate_after(a, before, tensor);
//   }
// ;


template<class A, class T>
concept asynchronous_allocator_of =
  allocator_of<A,T>

  and requires(A a)
  {
    {initial_happening(a)} -> happening;
  }

  and requires(A a, const initial_happening_result_t<A>& before, allocator_pointer_t<A,T> ptr, std::size_t n)
  {
    // XXX this needs to check that the result is a pair<happening,pointer>
    ubu::allocate_after<T>(a, before, n);
  
    {ubu::deallocate_after(a, before, fancy_span(ptr, n))} -> happening;
  }
;

template<class A>
concept asynchronous_allocator = allocator<A> and asynchronous_allocator_of<A,std::byte>;

} // end ubu

#include "../../../detail/epilogue.hpp"

