#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/concepts/decomposable.hpp"
#include "../../../../tensors/concepts/view.hpp"
#include "../../../../tensors/views/decompose.hpp"
#include "../../../../utilities/tuples.hpp"
#include "../concepts/allocator.hpp"
#include "custom_deallocate.hpp"

namespace ubu::detail
{

template<class A, class V>
concept has_decomposing_default_deallocate =
  view<V> and
  decomposable<V> and
  has_custom_deallocate<
    A,tuples::first_t<decompose_result_t<V>>
  >
;


template<class A, view V>
  requires has_decomposing_default_deallocate<A&&,V>
constexpr void decomposing_default_deallocate(A&& alloc, V tensor)
{
  // decompose the tensor into an underlying allocation and a layout, which is discarded
  auto [allocation, _] = decompose(tensor);

  return custom_deallocate(std::forward<A>(alloc), allocation);
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"


