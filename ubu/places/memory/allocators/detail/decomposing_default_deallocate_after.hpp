#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/concepts/decomposable.hpp"
#include "../../../../tensors/views/decompose.hpp"
#include "../../../../utilities/tuples.hpp"
#include "../concepts/allocator.hpp"
#include "custom_deallocate_after.hpp"

namespace ubu::detail
{

template<class A, class B, class T>
concept has_decomposing_default_deallocate_after =
  allocator<A>
  and happening<B>
  and decomposable<T>
  and has_custom_deallocate_after<
    A,B,tuples::first_t<decompose_t<T>>
  >
;


template<allocator A, happening B, tensor_like T>
  requires has_decomposing_default_deallocate_after<A&&,B&&,T&&>
constexpr happening auto decomposing_default_deallocate_after(A&& alloc, B&& before, T&& tensor)
{
  // decompose the tensor into an underlying allocation and a layout, which is discarded
  auto [allocation, _] = decompose(std::forward<T>(tensor));

  return custom_deallocate_after(std::forward<A>(alloc), std::forward<B>(before), std::move(allocation));
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

