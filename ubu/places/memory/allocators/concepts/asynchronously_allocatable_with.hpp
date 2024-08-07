#pragma once

#include "../../../../detail/prologue.hpp"

#include "../allocate_after.hpp"
#include <utility>

namespace ubu
{

template<class T, class A, class B, class S>
concept asynchronously_allocatable_with = requires(A alloc, B before, S shape)
{
  allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), std::forward<S>(shape));
};

} // end ubu

#include "../../../../detail/epilogue.hpp"

