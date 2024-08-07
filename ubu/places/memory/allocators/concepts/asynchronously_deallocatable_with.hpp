#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/concepts/tensor_like.hpp"
#include "../deallocate_after.hpp"
#include <utility>

namespace ubu
{


template<class D, class B, class T>
concept asynchronously_deallocatable_with = requires(D dealloc, B before, T tensor)
{
  deallocate_after(std::forward<D>(dealloc), std::forward<B>(before), std::forward<T>(tensor));
};


} // end ubu

#include "../../../../detail/epilogue.hpp"

