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

  and requires(A a, const initial_happening_result_t<A>& before, allocator_shape_t<A> shape)
  {
    allocate_after<T>(a, before, shape);
  }

  and requires(A a, allocate_after_result_t<T,A,initial_happening_result_t<A>,allocator_shape_t<A>> allocation)
  {
    deallocate_after(a, get<0>(allocation), get<1>(allocation));
  }
;

template<class A>
concept asynchronous_allocator = allocator<A> and asynchronous_allocator_of<A,std::byte>;

} // end ubu

#include "../../../../detail/epilogue.hpp"

