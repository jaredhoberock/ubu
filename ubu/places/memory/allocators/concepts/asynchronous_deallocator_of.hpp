#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/concepts/tensor_like.hpp"
#include "../deallocate_after.hpp"
#include <utility>

namespace ubu
{


template<class D, class T, class B>
concept asynchronous_deallocator_of =
  tensor_like<T>
  and happening<B>
  and requires(D dealloc, T tensor, B before)
  {
    deallocate_after(std::forward<D>(dealloc), std::forward<B>(before), std::forward<T>(tensor));
  }
;


} // end ubu

#include "../../../../detail/epilogue.hpp"

