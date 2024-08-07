#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include "../../../causality/initial_happening.hpp"
#include "../allocate_after.hpp"
#include "../deallocate_after.hpp"
#include "../traits/allocator_pointer.hpp"
#include "../traits/allocator_shape.hpp"
#include "../traits/allocator_value.hpp"
#include "allocator.hpp"
#include "asynchronously_allocatable_with.hpp"
#include "asynchronously_deallocatable_with.hpp"
#include <cstddef>
#include <memory>
#include <type_traits>

namespace ubu
{

template<class A, class T>
concept asynchronous_allocator_of =
  allocator_of<A,T>

  and requires(A a)
  {
    { initial_happening(a) } -> happening;
  }

  and asynchronously_allocatable_with<
    T,
    A,
    const initial_happening_result_t<A>&,
    allocator_shape_t<A>
  >

  and asynchronously_deallocatable_with<
    A, 
    tuples::first_t<allocate_after_result_t<T,A,initial_happening_result_t<A>,allocator_shape_t<A>>>,
    tuples::second_t<allocate_after_result_t<T,A,initial_happening_result_t<A>,allocator_shape_t<A>>>
  >
;

template<class A>
concept asynchronous_allocator = allocator<A> and asynchronous_allocator_of<A,std::byte>;

} // end ubu

#include "../../../../detail/epilogue.hpp"

