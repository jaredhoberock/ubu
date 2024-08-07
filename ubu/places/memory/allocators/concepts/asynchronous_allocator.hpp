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

// XXX S should default to A's allocator_shape_type, if it exists
template<class A, class T, class S = std::size_t>
concept asynchronous_allocator_of =
  std::is_object_v<T>

  and requires(A alloc)
  {
    { initial_happening(alloc) } -> happening;
  }

  and coordinate<S>
  and asynchronously_allocatable_with<T,A,initial_happening_result_t<A>,S>
  and asynchronously_deallocatable_with<
    A, 
    initial_happening_result_t<A>,
    tuples::second_t<allocate_after_result_t<T,A,initial_happening_result_t<A>,S>>
  >
;

template<class A>
concept asynchronous_allocator = allocator<A> and asynchronous_allocator_of<A,std::byte,allocator_shape_t<A>>;

} // end ubu

#include "../../../../detail/epilogue.hpp"

