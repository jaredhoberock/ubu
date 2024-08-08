#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/concepts/decomposable.hpp"
#include "../../../../tensors/concepts/view.hpp"
#include "../../../../tensors/views/decompose.hpp"
#include "../../../../utilities/tuples.hpp"
#include "../concepts/allocator.hpp"
#include "custom_deallocate_after.hpp"

namespace ubu::detail
{

template<class A, class B, class V>
concept has_decomposing_default_deallocate_after =
  allocator<A>
  and happening<B>
  and view<V>
  and decomposable<V>
  and has_custom_deallocate_after<
    A,B,tuples::first_t<decompose_result_t<V>>
  >
;


template<allocator A, happening B, view V>
  requires has_decomposing_default_deallocate_after<A&&,B&&,V>
constexpr happening auto decomposing_default_deallocate_after(A&& alloc, B&& before, V tensor)
{
  // decompose the tensor into an underlying allocation and a layout, which is discarded
  auto [allocation, _] = decompose(tensor);

  return custom_deallocate_after(std::forward<A>(alloc), std::forward<B>(before), allocation);
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

